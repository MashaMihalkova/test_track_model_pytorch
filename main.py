from model import *
import numpy as np
import pandas as pd
from dataloader import *
from enums import *

PATH_TO_PROJECTS = 'data/'
DOP_DATA_PATH = 'data/dop_materials/'
model_type = ModelType.Linear
criteria_type = CriteriaType.MAE
SAVE_WEIGHT = 'data/WEIGHTS/'
config = {
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 1,
        "num_workers": 0,
        "weight_decay(l2)": 0.05,
    }


def create_dataloaders_train_test(pd_data, model_type):  # noqa
    test_PD_DATA = pd_data.loc[(pd_data['month'] == 7) | (pd_data['month'] == 8)]
    test_PD_DATA = test_PD_DATA.loc[test_PD_DATA['year'] == 1]
    train_PD_DATA = pd_data[~pd_data.index.isin(test_PD_DATA.index)]
    train_PD_DATA = train_PD_DATA.sort_values(['year', 'month'])
    test_PD_DATA = test_PD_DATA.sort_values(by=['year'])

    train_dataset = PROJDataset(train_PD_DATA)  # noqa
    test_dataset = PROJDataset(test_PD_DATA)  # noqa

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)  # noqa
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)  # noqa
    return train_loader, test_loader


if __name__ == '__main__':
    Stat_PD_DATA = pd.read_excel(PATH_TO_PROJECTS + 'DATA.xlsx')  # noqa

    contr_id_real = list(Stat_PD_DATA['contr_id'].unique())
    uniq_contr = Stat_PD_DATA['contr_id'].unique()
    for i, contr in enumerate(uniq_contr):
        Stat_ = Stat_PD_DATA.loc[Stat_PD_DATA['contr_id'] == contr]
        if not Stat_.empty:
            Stat_['contr_id'] = i
    Stat_PD_DATA = Stat_
    contractor_id = list(Stat_PD_DATA['contr_id'].unique())

    year_unique = Stat_PD_DATA['year'].unique()
    Stat_PD_DATA_y = pd.DataFrame()
    for i, year in enumerate(year_unique):
        Stat_ = Stat_PD_DATA.loc[Stat_PD_DATA['year'] == year]
        if not Stat_.empty:
            Stat_['year'] = i
        Stat_PD_DATA_y = Stat_PD_DATA_y.append(Stat_)
    Stat_PD_DATA = Stat_PD_DATA_y

    uniq_month = Stat_PD_DATA['month'].unique()
    Stat_PD_DATA_m = pd.DataFrame()
    for i, month in enumerate(uniq_month):
        Stat_m = Stat_PD_DATA.loc[Stat_PD_DATA['month'] == month]
        if month > 12:
            Stat_m['month'] = month - 12
        Stat_PD_DATA_m = Stat_PD_DATA_m.append(Stat_m)
    Stat_PD_DATA = Stat_PD_DATA_m

    train_loader, test_loader = create_dataloaders_train_test(Stat_PD_DATA, model_type)
    feature_contractor_dict = np.load(DOP_DATA_PATH + 'all_contractors.npy', allow_pickle=True).item()
    stages_dict = np.load(DOP_DATA_PATH + 'stages.npy', allow_pickle=True).item()
    mech_res_dict = np.load(DOP_DATA_PATH + 'mech_res_dict.npy', allow_pickle=True).item()
    feature_contractor_dict_ids = {v: k for k, v in feature_contractor_dict.items()}
    stages_dict_ids = {v: k for k, v in stages_dict.items()}
    mech_res_ids = {v: k for k, v in mech_res_dict.items()}

    model_param = Parameters(config, model_type, criteria_type)
    model_param.net.load_state_dict(torch.load(f"{SAVE_WEIGHT}log_model_huber_05_epoch_500_loss_69_mae_547.pt"))

    tech = [2, 5, 8, 14, 19, 29, 30, 32, 42, 44, 46, 48, 57, 65, 70, 74, 76, 77, 83, 101, 111, 112, 115, 125,
            143, 157, 172, 209, 216, 234, 235]
    common_statist = []
    project_id = 23
    for ind, c in enumerate(contr_id_real):
        for i in tech:
            print(i)
            dict_statist = get_predict(Stat_PD_DATA, feature_contractor_dict, stages_dict, mech_res_dict,
                                       model_param.net, model_type, contractor_id=contractor_id[ind],
                                       contr_id_real=c, project_id=project_id,
                                       resource_id=i, print_result=True, plot_result=False)

            plt.show()
            if dict_statist != 0:
                common_statist.append(dict_statist)
