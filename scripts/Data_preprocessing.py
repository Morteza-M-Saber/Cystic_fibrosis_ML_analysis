import pandas as pd


class PROCESSOR:
    """A preprocessor for cystic fibrosis dataset

    Parameters
    ----------
    cf : pandas dataframe, shape = df[samples,features]
      Different cystic fibrosis dataset in pickled format

     target : str, shape = [n_classifiers], (default='FEVp_binary')
      target feature to be predicted

    fit_redundant : array-like, [fit1,fit2,...fitn],optional (default=['FEVp_binary','FEV_ranks_new_category',
                                'LungDecline','FEVp','Relative_Rate_of_lung_function_decline'])
      List of features equivalent or correlated with the labels to be predicted
      which needs to be eliminated from the training dataset
    """

    def __init__(self, cf, target, fit_redundant):
        self.cf = cf
        self.target = target
        self.fit_redundant = fit_redundant

    def preprocess(self, shannon_file, cystic_fibrosis_patient_cohort_data_path):
        """Process CYSTIC FIBROSIS data

        Args:
            shannon_file (Path): Path to meta data
            cystic_fibrosis_patient_cohort_data_path (Path): Path to genomic data

        Returns:
            DataFrame: Table of pre-processed genomic and metadata along with labels for ML prediction
        """
        df = pd.read_pickle(self.cf)
        ###@1-1)modifying the columns names and shortening them for better visualization.
        colName = df.columns
        colNameMod = [item.replace("Pseudomonas_aeruginosa_P749_PES", "PA749") for item in colName]
        df.columns = colNameMod
        ###@1-2) eliminating the phenotypes and the equivlaent columns
        for fit_ in self.fit_redundant:
            df.drop(fit_, axis=1, inplace=True)
        ###@1-4)converting snp frequencies to float
        df = df.astype(float)
        ###@1-3) converting the categorical columns which are defined by integer in the data.
        birth_cohort = {1.0: "a", 2.0: "b", 3.0: "c"}
        df["Birth_cohort"] = df["Birth_cohort"].map(birth_cohort)
        # adding shannon and simpson diversity to the data
        shan = pd.read_csv(shannon_file, encoding="unicode_escape")
        shan.set_index("SampleID", inplace=True)

        meta = pd.read_csv(cystic_fibrosis_patient_cohort_data_path)
        sample_map = {}
        name_ = meta.Sample_name
        id_ = meta.SampleID
        for _, i_ in enumerate(id_):
            sample_map[name_[i_].replace("IonCode_", "")] = id_[i_]

        shan_2add = []
        for samp_ in df.index:
            shan_2add.append(shan.loc[sample_map[samp_], "Shannon"])

        simp_2add = []
        for samp_ in df.index:
            simp_2add.append(shan.loc[sample_map[samp_], "Simpson"])

        Rate_of_lung_2_half_2add = []
        for samp_ in df.index:
            Rate_of_lung_2_half_2add.append(shan.loc[sample_map[samp_], "Rate_of_lung_2_half"])

        df["Shannon"] = shan_2add
        df["Simpson"] = simp_2add
        ###@1-4) OneHotEncoding for categorical data
        df = pd.get_dummies(df, drop_first=True)  # do one-hot-encoding using panda
        # Adding label for two-year-lung-decline
        df["Rate_of_lung_2_half"] = Rate_of_lung_2_half_2add
        # converting rapid:1 and nonrapid:0
        labMap = {"RP": 1, "Nrp": 0}
        df["Rate_of_lung_2_half"] = df["Rate_of_lung_2_half"].map(labMap)
        labels = df[self.target].values
        df.drop("Rate_of_lung_2_half", axis=1, inplace=True)  # drop the prediction label
        return (df, labels)
