import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.utils import Bunch

class LoadDataset:
    def __init__(self, which):
        if which == 'bc':
            self.data = load_breast_cancer()
        elif which == 'hp':
            df = pd.read_csv("data/hepatitis.csv")
            #df = df.fillna(method='ffill')
            #df['Sex'] = np.where(df['sex'] == 1, "M", "F")
            feature_names = list(df.columns)
            target_names = np.array(['yes', 'no'])

            data = np.array(df.iloc[:, 0:19])
            target = np.array(df['Class'])
            self.data = Bunch(data=data, target=target,
                              target_names=target_names,
                              feature_names=feature_names)
        elif which == 'ildp':
            df = pd.read_csv("data/ildp.csv")
            df = df.fillna(method='ffill')
            df['class'] = np.where(df['class'] == 'Yes', 0, 1)
            feature_names = list(df.columns)
            target_names = np.array(['yes', 'no'])

            data = np.array(df.iloc[:, 0:9])
            target = np.array(df['class'])

            self.data = Bunch(data=data, target=target,
                  target_names=target_names,
                  feature_names=feature_names)

        elif which == 'cv':
            df = pd.read_csv("data/covid19.csv")
            #df = df.fillna(method='ffill')
            #df.drop(columns=['patient_id']) # id not needed
            #df = df.drop('patient_id', 1) # id not needed
            df['sars_cov_2_exam_result'] = np.where(df['sars_cov_2_exam_result'] == 'negative', 0, 1)
            df['patient_addmited_to_regular_ward_1_yes_0_no'] = np.where(df['patient_addmited_to_regular_ward_1_yes_0_no'] == 'f', 0, 1)
            df['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] = np.where(df['patient_addmited_to_semi_intensive_unit_1_yes_0_no'] == 'f', 0, 1)
            df['patient_addmited_to_intensive_care_unit_1_yes_0_no'] = np.where(df['patient_addmited_to_intensive_care_unit_1_yes_0_no'] == 'f', 0, 1)
            #df['syncytial'] = np.where(df['syncytial'] == 'negative', 0, 1)
            # df['respiratory_syncytial_virus'] = np.where(df['respiratory_syncytial_virus'] == 'detected', 0, 1)
            # df['influenza_a'] = np.where(df['influenza_a'] == 'detected', 0, 1)
            # df['influenza_b'] = np.where(df['influenza_b'] == 'detected', 0, 1)
            # df['parainfluenza_1'] = np.where(df['parainfluenza_1'] == 'detected', 0, 1)
            # df['coronavirusnl63'] = np.where(df['coronavirusnl63'] == 'detected', 0, 1)
            # df['rhinovirus_enterovirus'] = np.where(df['rhinovirus_enterovirus'] == 'detected', 0, 1)
            # df['coronavirus_hku1'] = np.where(df['coronavirus_hku1'] == 'detected', 0, 1)
            # df['parainfluenza_3'] = np.where(df['parainfluenza_3'] == 'detected', 0, 1)
            # df['chlamydophila_pneumoniae'] = np.where(df['chlamydophila_pneumoniae'] == 'detected', 0, 1)
            # df['adenovirus'] = np.where(df['adenovirus'] == 'detected', 0, 1)
            # df['parainfluenza_4'] = np.where(df['parainfluenza_4'] == 'detected', 0, 1)
            # df['coronavirus229e'] = np.where(df['coronavirus229e'] == 'detected', 0, 1)
            # df['coronavirusoc43'] = np.where(df['coronavirusoc43'] == 'detected', 0, 1)
            # df['inf_a_h1n1_2009'] = np.where(df['inf_a_h1n1_2009'] == 'detected', 0, 1)
            # df['bordetella_pertussis'] = np.where(df['bordetella_pertussis'] == 'detected', 0, 1)
            # df['metapneumovirus'] = np.where(df['metapneumovirus'] == 'detected', 0, 1)
            # df['parainfluenza_2'] = np.where(df['parainfluenza_2'] == 'detected', 0, 1)
            # df['influenza_b_rapid_test'] = np.where(df['influenza_b_rapid_test'] == 'negative', 0, 1)
            # df['influenza_a_rapid_test'] = np.where(df['influenza_a_rapid_test'] == 'negative', 0, 1)
            # df['strepto_a'] = np.where(df['strepto_a'] == 'negative', 0, 1)
            # df['urine_esterase'] = np.where(df['urine_esterase'] == 'absent', 0, 1)
            # df['urine_ketone_bodies'] = np.where(df['urine_ketone_bodies'] == 'absent', 0, 1)
            # df['urine_urobilinogen'] = np.where(df['urine_urobilinogen'] == 'normal', 0, 1)
            # df['urine_protein'] = np.where(df['urine_protein'] == 'absent', 0, 1)
            # #df['urine_leukocytes'] = np.where(df['urine_leukocytes'] == '<1000', 1000)
            # df['urine_hyaline_cylinders'] = np.where(df['urine_hyaline_cylinders'] == 'absent', 0, 1)
            # df['urine_granular_cylinders'] = np.where(df['urine_granular_cylinders'] == 'absent', 0, 1)
            # df['urine_yeasts'] = np.where(df['urine_yeasts'] == 'absent', 0, 1)
            # df['urine_protein'] = np.where(df['urine_protein'] == 'absent', 0, 1)
            # df['urine_hemoglobin'] = np.where(df['urine_hemoglobin'] == 'present', 1, 0)
            # d = {'clear': 0, 'cloudy': 1, 'altered_coloring': 2, 'lightly_cloudy': 3,'Ausentes': 0,'Urato Amorfo - -+': 1,'Oxalato de Cálcio + ++': 2,'Oxalato de Cálcio - ++': 3,'Urato Amorfo + ++': 4,'light_yellow': 0, 'yellow':1, 'orange':2, 'citrus_yellow':3}
            # df = df.replace(d)

            #cleanup_nums = {"sars_cov_2_exam_result": {"negative": 0, "positive": 1},
                            # "patient_addmited_to_regular_ward_1_yes_0_no": {"negative": 0, "positive": 1},
                            # "patient_addmited_to_semi_intensive_unit_1_yes_0_no": {"negative": 0, "positive": 1},
                            # "patient_addmited_to_intensive_care_unit_1_yes_0_no": {"negative": 0, "positive": 1},
                            # "syncytial": {"negative": 0, "positive": 1},
                            # "respiratory_syncytial_virus": {"negative": 0, "positive": 1},
                            # "influenza_a": {"negative": 0, "positive": 1},
                            # "influenza_b": {"negative": 0, "positive": 1},
                            # "parainfluenza_1": {"negative": 0, "positive": 1},
                            # "coronavirusnl63": {"negative": 0, "positive": 1},
                            # "rhinovirus_enterovirus": {"negative": 0, "positive": 1},
                            # "coronavirus_hku1": {"negative": 0, "positive": 1},
                            # "parainfluenza_3": {"negative": 0, "positive": 1},
                            # "chlamydophila_pneumoniae": {"negative": 0, "positive": 1},
                            # "adenovirus": {"negative": 0, "positive": 1},
                            # "parainfluenza_4": {"negative": 0, "positive": 1},
                            # "coronavirus229e": {"negative": 0, "positive": 1},
                            # "coronavirusoc43": {"negative": 0, "positive": 1},
                            # "coronavirus_hku1": {"negative": 0, "positive": 1},
                            # "inf_a_h1n1_2009": {"negative": 0, "positive": 1},
                            # "bordetella_pertussis": {"negative": 0, "positive": 1},
                            # "metapneumovirus": {"negative": 0, "positive": 1},
                            # "parainfluenza_2": {"negative": 0, "positive": 1},
                            # "influenza_a_rapid_test": {"negative": 0, "positive": 1},
                            # "influenza_b_rapid_test": {"negative": 0, "positive": 1},
                            # "strepto_a": {"negative": 0, "positive": 1},
                            # }

            feature_names = list(df.columns)
            target_names = np.array(['yes', 'no'])

            df.fillna(0)
            data = np.array(df.iloc[:, 0:9])
            target = np.array(df['sars_cov_2_exam_result'])

            self.data = Bunch(data=data, target=target,
                  target_names=target_names,
                  feature_names=feature_names)
