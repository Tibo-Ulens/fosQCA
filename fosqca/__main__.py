import sys
import os
import argparse
import itertools
import copy

import numpy as np
import pandas as pd


class FosQca:
    def __init__(
        self,
        sets: list[pd.DataFrame],
        variables: list[str],
        outcome_col: str,
        caseid_col: str,
        outcome_value: int,
        consistency_threshold: float,
    ):
        self.sets = sets
        self.variables = variables
        self.outcome_col = outcome_col
        self.caseid_col = caseid_col
        self.outcome_value = int(outcome_value)
        self.consistency_threshold = consistency_threshold

        self.necessity_conditions = []
        self.variable_permutations = self.__generate_variable_permutations(variables)

    @classmethod
    def __generate_variable_permutations(cls, items: list):
        if len(items) == 0:
            return [[]]

        return cls.__generate_variable_permutations(items[1:]) + [
            [items[0]] + r for r in cls.__generate_variable_permutations(items[1:])
        ]

    def candidate_rules(self):
        """
        Generate all possible variable rules for all input sets
        """

        variable_permutations = self.variable_permutations

        rules = []

        # go through each permutation of the variables
        for i in range(len(variable_permutations)):
            print(f"=== generating rules for variables {variable_permutations[i]} ===")
            length = len(variable_permutations[i])

            # get the unique values for each variable
            per_column_unique_values = []
            for j in range(length):
                per_column_unique_values.append(
                    list(self.sets[0][variable_permutations[i][j]].unique())
                )

            print(f"unique values: {per_column_unique_values}")

            # get every permutation of the unique values so that all possible conditions can be
            # generated
            unique_values_permutations = [
                d for d in itertools.product(*per_column_unique_values)
            ]

            print(f"unique values permutations: {unique_values_permutations}")

            # generate a query for every unique value of every variable
            for r in range(len(unique_values_permutations)):
                condition_query = ""
                for j in range(length):
                    condition_query += (
                        f"`{variable_permutations[i][j]}`"
                        + "=="
                        + str(unique_values_permutations[r][j])
                        + " & "
                    )

                # trim trailing ' & '
                condition_query = condition_query[:-3]

                print(f"condition query: {condition_query}")

                # ignore queries that don't produce any results
                if (
                    condition_query == ""
                    or len(self.sets[0].query(condition_query)) == 0
                ):
                    print("no results for query\n")
                    continue

                # Get the rows matching the query
                result = self.sets[0].query(condition_query)
                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

                print(f"results for query:\n{result}\n")
                print(f"relative frequencies of outcome:\n{p}\n")

                # consistency = (# cases with condition and outcome) / (# cases with condition)
                # which is the same as the relative frequency of a 'correct' outcome in the result
                # column
                if (
                    p.idxmax() == self.outcome_value
                    and p[p.idxmax()] >= self.consistency_threshold
                ):
                    # query, cutoff, consistency
                    row = [
                        condition_query,
                        len(result[result[self.outcome_col] == self.outcome_value]),
                        p[p.idxmax()],
                    ]

                    rules.append(row)

            print("\n\n")

        rules.sort(key=lambda x: len(x[0]))
        rules.sort(key=lambda x: x[1], reverse=True)
        rules.sort(key=lambda x: x[2], reverse=True)
        print("There are {} candidate rules in total.".format(len(rules)))

        return rules

    # def search_necessity(self):
    #     # amount of rows where outcome_col == outcome_value
    #     cases_with_outcome = len(
    #         self.sets[0].loc[(self.sets[0][self.outcome_col] == self.outcome_value)]
    #     )

    #     print(f"issue: {cases_with_outcome}")

    #     if cases_with_outcome == 0:
    #         return

    #     necessity = dict()

    #     for variable in self.variables:
    #         for value in self.sets[0][variable].unique():
    #             consistency = (
    #                 len(
    #                     self.sets[0].loc[
    #                         (self.sets[0][self.outcome_col] == self.outcome_value)
    #                         & (self.sets[0][variable] == float(value))
    #                     ]
    #                 )
    #                 / cases_with_outcome
    #             )

    #             print(f"consistency for {variable}={value} :: {consistency}")

    #             if consistency >= self.consistency_threshold:
    #                 print("{}=={} is a necessity condition".format(variable, value))

    #                 necessity[variable] = value
    #                 self.necessity_conditions.append(
    #                     f"`{variable}`" + "==" + str(value)
    #                 )

    # def __check_subset(self, new_rule, rules, unique_cover):
    #     final_rules = copy.deepcopy(rules)
    #     final_rules.append(new_rule)
    #     rules = []
    #     set_A = set()

    #     for i in range(len(final_rules)):
    #         set_B = set()
    #         for j in range(i + 1, len(final_rules)):
    #             temp = self.sets[0].query(final_rules[j])
    #             index = set(
    #                 temp[temp[self.outcome_col] == self.outcome_value].index.tolist()
    #             )
    #             set_B = set_B.union(index)
    #             # temp[self.outcome_col].value_counts(normalize=False, dropna=True)

    #         temp = self.sets[0].query(final_rules[i])
    #         index = set(
    #             temp[temp[self.outcome_col] == self.outcome_value].index.tolist()
    #         )

    #         if len(index.difference(set_B.union(set_A))) < unique_cover:
    #             pass
    #         else:
    #             rules.append(final_rules[i])
    #             set_A = set_A.union(index)

    #     return rules, set_A

    # def greedy(self, rules, unique_cover=1):
    #     if len(rules) == 0:
    #         print("The candidate rule list is empty.")
    #         return [], set()

    #     final_set = set()
    #     final_rules = []
    #     for i in range(len(rules)):
    #         print(f"'checking' rule {rules[i][0]}")

    #         flag = False
    #         for necessity_condition in self.necessity_conditions:
    #             if necessity_condition in rules[i][0]:
    #                 flag = True

    #         if flag:
    #             continue

    #         temp_final_rule, temp_set = self.__check_subset(
    #             rules[i][0], final_rules, unique_cover
    #         )

    #         if len(temp_set) > len(final_set):
    #             final_rules, final_set = temp_final_rule, temp_set

    #     if len(final_rules) == 0:
    #         return [], set()

    #     for i in range(len(final_rules)):
    #         for j in range(len(self.necessity_conditions)):
    #             final_rules[i] = final_rules[i] + " & " + self.necessity_conditions[j]

    #     final_set = set()
    #     for rule in final_rules:
    #         cases = self.sets[0].query(rule)
    #         final_set = final_set.union(
    #             set(list(cases[cases[self.outcome_col] == self.outcome_value].index))
    #         )

    #     return final_rules, final_set


if __name__ == "__main__":
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--outcome-col", default="outcome")
    parser.add_argument("-c", "--caseid-col", default="case")
    parser.add_argument("-v", "--outcome-value", default=1, type=int)
    parser.add_argument("--consistency", default=0.8, type=float)
    parser.add_argument("sets", nargs="*")

    args = parser.parse_args()

    for set_file in args.sets:
        if not os.path.isfile(set_file):
            print(f"'{set_file}' is not a file")
            sys.exit(1)

    sets = []
    for set_file in args.sets:
        data = pd.read_csv(set_file)
        data.attrs = {"file": set_file}

        sets.append(data)

    cols = sets[0].columns
    for s in sets[1:]:
        if (s.columns != cols).any():
            print(f"{s.attrs["file"]} has invalid headers, expected {list(cols)}")

    cols = list(cols)

    if args.outcome_col in cols:
        cols.remove(args.outcome_col)
    if args.caseid_col in cols:
        cols.remove(args.caseid_col)

    qca = FosQca(
        sets,
        variables=cols,
        outcome_col=args.outcome_col,
        caseid_col=args.caseid_col,
        outcome_value=args.outcome_value,
        consistency_threshold=args.consistency,
    )

    rules = qca.candidate_rules()
    rules = pd.DataFrame(
        rules, columns=["candidate_rule", "cutoff", "consistency"]
    ).sort_values(by=["cutoff", "consistency"], ascending=False)

    print(rules)
    print()

    # qca.search_necessity()
    # print(f"necessity conditions: {qca.necessity_conditions}")

    # config, sets = qca.greedy(rules.values.tolist())

    # print(config)
    # print(sets)
