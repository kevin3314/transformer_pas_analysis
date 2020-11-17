def case_analysis_f1_ga(result: dict):
    return result['ガ']['case_analysis'].f1


def case_analysis_f1_wo(result: dict):
    return result['ヲ']['case_analysis'].f1


def case_analysis_f1_ni(result: dict):
    return result['ニ']['case_analysis'].f1


def case_analysis_f1_ga2(result: dict):
    return result['ガ２']['case_analysis'].f1


def case_analysis_f1(result: dict):
    return result['all_case']['case_analysis'].f1


def zero_anaphora_f1_ga(result: dict):
    return result['ガ']['zero'].f1


def zero_anaphora_f1_wo(result: dict):
    return result['ヲ']['zero'].f1


def zero_anaphora_f1_ni(result: dict):
    return result['ニ']['zero'].f1


def zero_anaphora_f1_ga2(result: dict):
    return result['ガ２']['zero'].f1


def zero_anaphora_f1(result: dict):
    return result['all_case']['zero'].f1


def zero_anaphora_f1_inter(result: dict):
    return result['all_case']['zero_inter'].f1


def zero_anaphora_f1_intra(result: dict):
    return result['all_case']['zero_intra'].f1


def zero_anaphora_f1_exophora(result: dict):
    return result['all_case']['zero_exophora'].f1


def coreference_f1(result: dict):
    return result['all_case']['coreference'].f1


def bridging_anaphora_f1(result: dict):
    return result['all_case']['bridging'].f1
