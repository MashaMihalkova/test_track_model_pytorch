import torch
import torch.nn as nn


class positive_weights_linear(nn.Module):
    """
        class with positive weights output
        input array (shape (373) ) and res_id (torch.int64 or torch.long)
    """

    def __init__(self, in_features, out_features):
        super(positive_weights_linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input_x, res_id):
        return torch.matmul(self.weight[res_id.long()].exp(), input_x)


class Model_predict_hours_net(nn.Module):
    """
        всего работ по нормам 373 шт. = "in_features" в слое "activity_dense"
        всего видов техники = 246 (все виды техники берутся из норм PO№№№ ) = "out_features" в слое "activity_dense"
        всего подрядчиков 4 шт. = "in_features" в слое "contractor_dense"
        всего проектов 23 шт. = "in_features" в слое "proj_dense"
    """

    def __init__(self, input_size, hidden_size, num_layers, device):
        super(Model_predict_hours_net, self).__init__()
        #  out_features = количество техник 246
        self.activity_dense = positive_weights_linear(in_features=373, out_features=246)
        self.proj_dense = torch.nn.Linear(in_features=23, out_features=1, bias=False)
        self.contractor_dense = torch.nn.Linear(in_features=4, out_features=1, bias=False)
        self.year_dense = torch.nn.Linear(in_features=2, out_features=1, bias=True)
        self.month_dense = torch.nn.Linear(in_features=12, out_features=1, bias=False)

    def forward(self, x):
        # умножили на матрицу весов
        sum_of_activities = self.activity_dense(x[2:-3], x[-1].long())

        # умножили на коэф месяца
        sum_of_activities_month = self.month_dense.weight[0, x[-3].long()] * sum_of_activities

        # умножили на коэф.года
        sum_of_activities_month_year = self.year_dense.weight[0, x[-2].long()] * sum_of_activities_month

        # умножили на коэф.контрактора
        predict = self.contractor_dense.weight[0, x[1].long()] * sum_of_activities_month_year
        return predict
