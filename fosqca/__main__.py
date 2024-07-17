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

    def generate_query(self, abstract_query: list):
        """
        Generate a textual query from an abstract one
        """

        query = ""

        for i in range(len(abstract_query)):
            if abstract_query[i] is None:
                continue

            query += f"`{self.variables[i]}`" + "==" + str(abstract_query[i]) + "&"

        return query[:-1]

    def generate_rules(self) -> pd.DataFrame:
        """
        Generate all possible variable rules for all input sets
        """

        rules = []

        # get all the unique values each variable can assume
        per_variable_unique_values = []
        for i in range(len(self.variables)):
            per_variable_unique_values.append(
                list(self.sets[0][self.variables[i]].unique())
            )

        # generate every permutation of these unique values
        unique_values_permutations = [
            p for p in itertools.product(*per_variable_unique_values)
        ]

        for value_permutation in unique_values_permutations:
            query = self.generate_query(value_permutation)

            # Get the rows matching the query
            result = self.sets[0].query(query)

            if result.empty:
                row = [query, value_permutation, 0, 0, -1.0]
                rules.append(row)

                continue

            # Get the relative frequencies of the values of the outcome column
            p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

            print(f"results for query {query}:\n{result}\n")
            print(f"relative frequencies of outcome:\n{p}\n")

            # consistency = (# cases with condition and outcome) / (# cases with condition)
            # which is the same as the relative frequency of a 'correct' outcome in the result
            # column
            if p.get(self.outcome_value, 0.0) >= self.consistency_threshold:
                row = [
                    query,
                    value_permutation,
                    len(result[result[self.outcome_col] == self.outcome_value]),
                    len(result[result[self.outcome_col] != self.outcome_value]),
                    p.get(self.outcome_value, 0.0),
                ]

                rules.append(row)

        print(f"generated {len(rules)} candidate rules")

        rules = pd.DataFrame(
            rules,
            columns=[
                "rule",
                "values",
                "positive_cases",
                "negative_cases",
                "consistency",
            ],
        )

        return rules

    @staticmethod
    def can_be_merged(rulea: list, ruleb: list) -> (bool, int):
        distance = 0
        idx = 0

        for i, (a, b) in enumerate(zip(rulea, ruleb)):
            if a != b:
                distance += 1
                idx = i

        if distance != 1:
            return (False, idx)

        return (True, idx)

    def merge_rule_list(self, rules: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Attempt to merge rules in a given list, returns the new list of rules
        and a list of the unmerged rules that should be retained
        """

        merged_rules = set()
        new_rules = []

        for i in range(len(rules)):
            rulea = rules.iloc[i]

            for j in range(i + 1, len(rules)):
                ruleb = rules.iloc[j]

                cond_distance = self.can_be_merged(
                    rulea.get("values"), ruleb.get("values")
                )
                if not cond_distance[0]:
                    continue

                cons = (rulea.get("consistency"), ruleb.get("consistency"))

                match cons:
                    case (
                        x,
                        y,
                    ) if x >= self.consistency_threshold and y >= self.consistency_threshold:
                        pass
                    case (x, -1.0) if x >= self.consistency_threshold:
                        pass
                    case (-1.0, x) if x >= self.consistency_threshold:
                        pass
                    case (-1.0, -1.0):
                        pass
                    case _:
                        continue

                new_rule_values = copy.deepcopy(rulea.get("values"))
                new_rule_values = list(new_rule_values)
                new_rule_values[cond_distance[1]] = None
                new_rule_values = tuple(new_rule_values)

                new_query = self.generate_query(new_rule_values)

                rulea_values = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(rulea.get("values")),
                    )
                )
                ruleb_values = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(ruleb.get("values")),
                    )
                )
                new_values_pretty = list(
                    map(
                        lambda x: int(x) if x is not None else None,
                        list(new_rule_values),
                    )
                )
                print(
                    f"merged queries {rulea_values} {rulea.get("consistency")} + {ruleb_values} {ruleb.get("consistency")} -> {new_values_pretty}\n"
                )

                merged_rules.add(rulea.get("rule"))
                merged_rules.add(ruleb.get("rule"))

                # Get the rows matching the query
                result = self.sets[0].query(new_query)

                if result.empty:
                    row = [new_query, new_rule_values, 0, 0, -1.0]
                    new_rules.append(row)

                    continue

                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

                # print(f"results for query {new_query}:\n{result}\n")
                # print(f"relative frequencies of outcome:\n{p}\n")

                # consistency = (# cases with condition and outcome) / (# cases with condition)
                # which is the same as the relative frequency of a 'correct' outcome in the result
                # column
                if p.get(self.outcome_value, 0.0) >= self.consistency_threshold:
                    row = [
                        new_query,
                        new_rule_values,
                        len(result[result[self.outcome_col] == self.outcome_value]),
                        len(result[result[self.outcome_col] != self.outcome_value]),
                        p.get(self.outcome_value, 0.0),
                    ]

                    new_rules.append(row)

        new_rules = pd.DataFrame(
            new_rules,
            columns=[
                "rule",
                "values",
                "positive_cases",
                "negative_cases",
                "consistency",
            ],
        )

        unmerged_rules = rules[
            rules.apply(lambda r: r.get("rule") not in merged_rules, axis=1)
        ]

        return (new_rules, unmerged_rules)

    def merge_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        new_rules, unmerged_rules = self.merge_rule_list(rules)
        rules = pd.concat([new_rules, unmerged_rules])

        while True:
            new_rules, unmerged_rules = self.merge_rule_list(rules)

            if new_rules.empty:
                break

            rules = pd.concat([new_rules, unmerged_rules])
            rules = rules.drop_duplicates()

            print(f"new rules:\n{rules}")

        return rules

    def get_minimal_necessary_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        """
        Get the minimal number of rules that cover all positive outcomes
        """

        rules = rules.sort_values(by=["positive_cases"], ascending=False)

        necessary_cases = set(
            self.sets[0][
                self.sets[0][self.outcome_col] == self.outcome_value
            ].index.to_list()
        )
        covered_cases = set()
        necessary_rules = []

        for idx, rule in rules.iterrows():
            # Get the rows matching the rule
            result = qca.sets[0].query(rule.get("rule"))
            covered_cases.update(result.index.to_list())
            necessary_rules.append(rule)

            if covered_cases == necessary_cases:
                break

        necessary_rules = pd.DataFrame(
            necessary_rules,
            columns=[
                "rule",
                "values",
                "positive_cases",
                "negative_cases",
                "consistency",
            ],
        )

        return necessary_rules


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

    rules = qca.generate_rules()

    print(f"possible rules:\n{rules}\n")

    merged_rules = qca.merge_rules(rules)

    print(f"merged rules:\n{merged_rules}\n")

    necessary_rules = qca.get_minimal_necessary_rules(merged_rules)

    print(f"necessary rules:\n{necessary_rules}\n")
