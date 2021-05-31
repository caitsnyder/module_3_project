import numpy as np
import pandas as pd

from fuzzywuzzy import fuzz
from fuzzywuzzy import process


class FuzzyMatcher:
    def __init__(self):
        self.cols = []
    
    def set_fuzzy_matches(self, df, cols):
        self.cols = cols
        self.clean_col_text(df)
        
        scores_df = self.get_matches(df)       
        self.replace_with_matches(df, scores_df) 
        
    def clean_col_text(self, df):
        for col in self.cols:
            df[col] = df[col].str.lower()
            df[col] = df.apply(lambda row: self.manually_clean_col_text(str(row[col])), axis=1)
            df[col] = df.apply(lambda row: self.remove_special_chars(str(row[col])), axis=1)
            
            
    def manually_clean_col_text(self, value):
        replace_dict = { # ideally in constants file
            "private individual": "private",
            "not known": "unknown",
            "0": "unknown",
            "-": "unknown",
            "nan": "unknown",
            "action in a": "action in africa",
            "wateraid": "water aid"
        }
        if value in replace_dict.keys():
            value = replace_dict[value]
        return value
    
    def remove_special_chars(self, value):
        special_chars = [".", "/", "-", "[", "]", "(", ")"]
        for char in special_chars:
            value = value.replace(char, "")
        return value
    
    def get_matches(self, df):
        values = self.get_values(df)
        match_df = self.get_match_df(values)
        scores_df = self.get_scores(match_df)
        return scores_df

    def get_values(self, df):
        list_of_lists = [df[col] for col in self.cols]
        values = [j for sub in list_of_lists for j in sub]
        return np.unique([str(i).lower() for i in values if str(i) != 'nan']).tolist()
    
    def get_match_df(self, values):
        score_sort = [(x,) + i
                     for x in values 
                     for i in process.extract(x, values, scorer=fuzz.token_sort_ratio)]
        match_df = pd.DataFrame(score_sort, columns=['name_sort','match_sort','score_sort'])
        match_df['sorted_name_sort'] = \
            np.minimum(match_df['name_sort'], match_df['match_sort'])
        return match_df
    
    def get_scores(self, match_df):
        high_score_sort = self.get_score_sort(match_df)
        scores = self.get_score_groups(high_score_sort)
        scores_df = self.get_score_frame(scores)
        return scores_df

    def get_score_sort(self, match_df):
        high_score_sort = \
            match_df.loc[(match_df['score_sort'] >= 80) &
                    (match_df['name_sort'] !=  match_df['match_sort']) &
                    (match_df['sorted_name_sort'] != match_df['match_sort'])]
        
        high_score_sort = high_score_sort.drop('sorted_name_sort', axis=1).copy()
        return high_score_sort
        

    def get_score_groups(self, high_score_sort):
        return high_score_sort.groupby(['name_sort','score_sort']).agg(
                        {'match_sort': ', '.join}).sort_values(
                        ['score_sort'], ascending=False)
        
    def get_score_frame(self, scores):
        frame = { 
            'name_sort': scores.index.get_level_values(0), 
            'score_sort': scores.index.get_level_values(1),
            'match_sort': list(map(lambda x: x[0], scores.values.tolist())),
        }
        return pd.DataFrame(frame)
    
    def replace_with_matches(self, df, df_matches):
        for col in self.cols:
            for name in df_matches["name_sort"]:
                if name in df[col].values.tolist():
                    replace_value = df_matches.loc[df_matches["name_sort"] == name,
                                                   "match_sort"].values.tolist()[0]
                    df[col].replace(name, replace_value, inplace=True)

