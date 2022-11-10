import torch
from matplotlib import pyplot as plt


# предикт по всем данным
def show_results(y_true, y_pred, sum_plans, name=None):
    plt.plot(y_true, label='Целевое значение')
    plt.plot(y_pred, label='Прогноз ИИ модели', color='orange')
    plt.plot(sum_plans, label='Плановое значение Primavera', color='g')

    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('месяцы')
    plt.ylabel('машиночасы')
    if not name: name = 'Cравнение применения ИИ моделей для прогнозирования техники'
    plt.title(name);


def get_predict(PD_DATA, feature_contractor_dict_ids, stages_dict_ids, mech_res_ids, model, model_type, contractor_id,
                contr_id_real, project_id, resource_id, print_result=False, plot_result=True):
    a = list(map(str, range(0, 373)))
    b = ['proj_id', 'contr_id']
    b.extend(a)
    # if model_type is ModelType.Linear_3MONTH:
    #     b.extend(['month', 'year', 'res_id', 'm_1', 'm_2', 'm_3', 'target'])
    # else:
    b.extend(['month', 'year', 'res_id', 'target'])
    df_for_predict = PD_DATA.loc[
        (PD_DATA.contr_id == contractor_id) & (PD_DATA.proj_id == project_id) & (PD_DATA.res_id == resource_id), b]

    features_for_predict = torch.tensor(df_for_predict.iloc[:, :-1].values).to(torch.float)
    target = df_for_predict.iloc[:, -1].values

    plans = df_for_predict.iloc[:, 2:-4]

    sum_plans = plans.sum(axis=1)

    sum_plans = sum_plans.values

    predict = []

    with torch.no_grad():
        for month in range(features_for_predict.shape[0]):
            a = model(features_for_predict[month])
            predict.append(a.tolist())

    if print_result:
        if target.shape[0] != 0:
            plan_targ_predict = {'resource': [], 'month': [], 'predict': [], 'target': [], 'plan': []}
            plan_targ_predict['resource'].append(list(mech_res_ids.keys())[list(mech_res_ids.values()).index(resource_id)])
            for i in range(len(predict)):
                # print(f'month = {i+1}, predict = {predict[i] :.2f}, target = {target[i] :.2f}')
                plan_targ_predict['month'].append(i)
                plan_targ_predict['predict'].append(predict[i])
                plan_targ_predict['target'].append(target[i])
                plan_targ_predict['plan'].append(sum_plans[i])
            print(plan_targ_predict)
            return plan_targ_predict
        else:
            return 0

    if plot_result:
        if target.shape[0] != 0:
            print(f'Resource_id = {resource_id}')
            Contractor = list(feature_contractor_dict_ids.keys())[list(feature_contractor_dict_ids.values()).index(contr_id_real)]
            project = list(stages_dict_ids.keys())[list(stages_dict_ids.values()).index(project_id)]
            resource = list(mech_res_ids.keys())[list(mech_res_ids.values()).index(resource_id)]

            show_results(target, predict, sum_plans,
                         name=f'Contractor = {Contractor},'
                              f' project = {project} \n resource = {resource}')
        return 0