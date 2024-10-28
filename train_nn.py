import torch
from torch import nn, optim
from torch.nn import functional as F
from chess_game import ChessGame
import numpy as np
import os

nn_args: dict = {
         'lr': 0.02,
        'dropout': 0.3,
        'epochs': 20,
        'batch_size': 64,
        'cuda': True,
        'num_channels': 128,
    }


class ChessNN(nn.Module):
    optimizer: optim.Optimizer
    learning_rate: float
    action_size: int
    game: ChessGame
    hyperparam_args: dict = nn_args

    def __init__(self, game: ChessGame):
        super(ChessNN, self).__init__()
        self.game = game
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.create_neural_net()
        self.learning_rate = 1e-3
        self.optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)

    def create_neural_net(self):
        args = self.hyperparam_args
        num_channels = args["num_channels"]

        # Input Layer
        self.conv1 = nn.Conv3d(1, num_channels, 3, stride=1,padding=1)

        # Hidden Layer
        self.conv2 = nn.Conv3d(num_channels, num_channels * 2, 3, stride=1, padding = 1)
        self.conv3 = nn.Conv3d(num_channels*2,num_channels*2,3,stride=1)
        self.conv4 = nn.Conv3d(num_channels*2,num_channels*2,3,stride=1)
        self.conv5 = nn.Conv3d(num_channels*2, num_channels, 1, stride = 1)

        self.bn1 = nn.BatchNorm3d(num_channels)
        self.bn2 = nn.BatchNorm3d(num_channels * 2)
        self.bn3 = nn.BatchNorm3d(num_channels * 2)
        self.bn4 = nn.BatchNorm3d(num_channels * 2)
        self.bn5 = nn.BatchNorm3d(num_channels)

        self.fc1 = nn.Linear(num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4), 1024) #4096 -> 1024
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 512)
        self.fc_bn3 = nn.BatchNorm1d(512)

        # output p dist        
        self.fc4 = nn.Linear(512, self.action_size)

        # output scalar
        self.fc5 = nn.Linear(512, 1)

    def forward(self, board, validMoves): # Get Best next Move
        dropout = self.hyperparam_args["dropout"]
        num_channels = self.hyperparam_args["num_channels"]


        board = board.view(-1, 1, self.board_x, self.board_y, self.board_z)
        board = F.relu(self.bn1(self.conv1(x)))
        board = F.relu(self.bn2(self.conv2(x)))
        board = F.relu(self.bn3(self.conv3(x)))
        board = F.relu(self.bn4(self.conv4(x)))
        board = F.relu(self.bn5(self.conv5(x)))
        board = board.view(-1, num_channels*(self.board_x-4)*(self.board_y-4)*(self.board_z-4))
        board = F.dropout(F.relu(self.fc_bn1(self.fc1(x))), p=dropout, training=self.training)
        board = F.dropout(F.relu(self.fc_bn2(self.fc2(x))), p=dropout, training=self.training)
        board = F.dropout(F.relu(self.fc_bn3(self.fc3(x))), p=dropout, training=self.training)

        pi = self.fc4(board)
        v = self.fc5(board)

        pi -= (1 - validMoves) * 1000
        return F.log_softmax(pi, dim=1), torch.tanh(v)


class ChessEvaluator(object):

    neural_net: ChessNN
    action_size: int
    board_x: int
    board_y: int
    board_z: int

    def __init__(self, game: ChessGame):
        super(ChessEvaluator, self).__init__()

        self.neural_net = ChessNN(game)
        self.board_x, self.board_y, self.board_z = game.getBoardSize()
        self.action_size = game.getActionSize()

        self.neural_net.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.neural_net.parameters())
        batch_size = nn_args["batch_size"]

        for epoch in range(nn_args["epochs"]):
            
            print(f"EPOCH :: f{str(epoch+1)}")
            self.neural_net.train()
            idx = 0

            while(idx < int(len(examples)) / batch_size):
                sample_indices = np.random.randint(len(examples), size=batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_indices]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                target_valids = torch.FloatTensor(np.array(valids))

                boards, target_pis, target_vs, target_valids = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda(), target_valids.contiguous().cuda()

                # compute output
                out_pi, out_v = self.nnet((boards, target_valids))
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                
                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, boardAndValid) -> tuple:
        board, validMoves = boardAndValid

        # preparing input
        board = torch.FloatTensor(board.astype(np.float64))
        validMoves = torch.FloatTensor(validMoves.astype(np.float64))
        if nn_args["cuda"]:
            board = board.contiguous().cuda()
            validMoves = validMoves.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y, self.board_z)
        
        # predict without changing weigths
        self.neural_net.eval()
        with torch.no_grad():
            pi, v = self.neural_net((board, validMoves))

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """Custom loss function for probabilty distribuition"""
        return -torch.sum(targets*outputs)/targets.size()[0]

    def loss_v(self, targets, outputs):
        """Custom loss function for scalar value"""
        return torch.sum((targets-outputs.view(-1))**2)/targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        """Save weights checkpoint"""
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")

        torch.save({
            'state_dict' : self.neural_net.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar') -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))

        map_location = None if nn_args["cuda"] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.neural_net.load_state_dict(checkpoint['state_dict'])

    def print(self, game) -> None:
        """Print current self object state"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(ChessNN(game, nn_args).to(device))



