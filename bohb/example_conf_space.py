import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_configspace():
    """
    It builds the configuration space with the needed hyperparameters.
    It is easily possible to implement different types of hyperparameters.
    Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
    :return: ConfigurationsSpace-Object
    """
    cs = CS.ConfigurationSpace()

    lr = CSH.UniformFloatHyperparameter('optParam.lr', lower=1e-6, upper=1e-1, default_value='0.00002', log=True)
    weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=1e-6, upper=0.1, default_value=1e-3, log=True)

    # For demonstration purposes, we add different optimizers as categorical hyperparameters.
    # To show how to use conditional hyperparameters with ConfigSpace, we'll add the optimizers 'Adam' and 'SGD'.
    # SGD has a different parameter 'momentum'.
    optimizer = CSH.CategoricalHyperparameter('optType', ['AdamW', 'Ranger'])

    rgr_K = CSH.UniformIntegerHyperparameter('optParam.k', lower=4, upper=9, default_value=7, log=False)

    cs.add_hyperparameters([lr, optimizer, weight_decay, rgr_K])

    # The hyperparameter rgr_K will be used,if the configuration
    # contains 'Ranger' as optimizer.
    cond = CS.EqualsCondition(rgr_K, optimizer, 'Ranger')
    cs.add_condition(cond)

    return cs
