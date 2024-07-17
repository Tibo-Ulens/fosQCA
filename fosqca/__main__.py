import sys
import os
import argparse
import itertools
import copy
import logging

import numpy as np
import pandas as pd


logger = logging.getLogger("fosQCA")


class FosQca:
    def __init__(
        self,
        sets: dict[str, pd.DataFrame],
        variables: list[str],
        outcome_col: str,
        outcome_value: int,
        consistency_threshold: float,
        coverage_threshold: float,
    ):
        self.sets = sets
        self.variables = variables
        self.outcome_col = outcome_col
        self.outcome_value = int(outcome_value)
        self.consistency_threshold = consistency_threshold
        self.coverage_threshold = coverage_threshold

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

    def generate_rules(self) -> dict[str, pd.DataFrame]:
        """
        Generate all possible variable rules for all input sets
        """

        rules = dict()

        for filename, dataset in self.sets.items():
            dataset_rules = []

            # get all the unique values each variable can assume
            per_variable_unique_values = []
            for i in range(len(self.variables)):
                per_variable_unique_values.append(
                    list(dataset[self.variables[i]].unique())
                )

            # generate every permutation of these unique values
            unique_values_permutations = [
                p for p in itertools.product(*per_variable_unique_values)
            ]

            for value_permutation in unique_values_permutations:
                query = self.generate_query(value_permutation)

                # Get the rows matching the query
                result = dataset.query(query)

                if result.empty:
                    row = [query, value_permutation, 0, 0, -1.0]
                    dataset_rules.append(row)

                    continue

                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

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

                    dataset_rules.append(row)

            dataset_rules = pd.DataFrame(
                dataset_rules,
                columns=[
                    "rule",
                    "values",
                    "positive_cases",
                    "negative_cases",
                    "consistency",
                ],
            )

            dataset_rules = dataset_rules.drop_duplicates()

            rules[filename] = dataset_rules

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

    def merge_rule_list(
        self, rules: pd.DataFrame, dataset_name: str
    ) -> (pd.DataFrame, pd.DataFrame):
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

                logger.debug(
                    f"merged queries {rulea_values} {rulea.get("consistency")} + {ruleb_values} {ruleb.get("consistency")} -> {new_values_pretty}\n"
                )

                merged_rules.add(rulea.get("rule"))
                merged_rules.add(ruleb.get("rule"))

                # Get the rows matching the query
                # result = self.sets[0].query(new_query)
                result = self.sets[dataset_name].query(new_query)

                if result.empty:
                    row = [new_query, new_rule_values, 0, 0, -1.0]
                    new_rules.append(row)

                    continue

                # Get the relative frequencies of the values of the outcome column
                p = result[self.outcome_col].value_counts(normalize=True, dropna=False)

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

    def merge_rules(self, rules: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        new_rules, unmerged_rules = self.merge_rule_list(rules, dataset_name)
        rules = pd.concat([new_rules, unmerged_rules])

        while True:
            new_rules, unmerged_rules = self.merge_rule_list(rules, dataset_name)

            if new_rules.empty:
                break

            rules = pd.concat([new_rules, unmerged_rules])
            rules = rules.drop_duplicates()

            logger.debug(f"new rules:\n{rules}\n")

        return rules

    def get_minimal_necessary_rules(
        self, rules: pd.DataFrame, dataset_name: str
    ) -> pd.DataFrame:
        """
        Get the minimal set of rules required to reach the coverage threshold
        """

        rules = rules.sort_values(by=["positive_cases"], ascending=False)

        necessary_cases = set(
            self.sets[dataset_name][
                self.sets[dataset_name][self.outcome_col] == self.outcome_value
            ].index.to_list()
        )

        covered_cases = set()
        necessary_rules = []

        for idx, rule in rules.iterrows():
            # Get the rows matching the rule
            result = self.sets[dataset_name].query(rule.get("rule"))
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
    parser.add_argument("-l", "--outcome-col", default="outcome")
    parser.add_argument("-o", "--outcome-value", default=1, type=float)
    parser.add_argument("-i", "--ignore-col", action="append")
    parser.add_argument("--verbose", action="count", default=0)
    parser.add_argument("-c", "--consistency", default=0.8, type=float)
    parser.add_argument("-v", "--coverage", default=0.8, type=float)
    parser.add_argument("sets", nargs="*")

    args = parser.parse_args()

    match args.verbose:
        case 0:
            verbosity = logging.INFO
        case _:
            verbosity = logging.DEBUG

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.setLevel(verbosity)
    logger.addHandler(handler)

    if len(args.sets) < 1:
        print("at least 1 dataset is required")
        sys.exit(1)

    for set_file in args.sets:
        if not os.path.isfile(set_file):
            print(f"'{set_file}' is not a file")
            sys.exit(1)

    sets = dict()
    cols = None
    for set_file in args.sets:
        data = pd.read_csv(set_file)
        data.attrs = {"file": set_file}

        sets[set_file] = data
        cols = data.columns

    for filename, s in sets.items():
        if (s.columns != cols).any():
            print(f"{filename} has invalid headers, expected {list(cols)}")

    cols = list(cols)

    if args.outcome_col in cols:
        cols.remove(args.outcome_col)

    for ignored in args.ignore_col:
        if ignored in cols:
            cols.remove(ignored)

    qca = FosQca(
        sets,
        variables=cols,
        outcome_col=args.outcome_col,
        outcome_value=args.outcome_value,
        consistency_threshold=args.consistency,
        coverage_threshold=args.coverage,
    )

    rules = qca.generate_rules()

    for filename, file_rules in rules.items():
        logger.info(f"possible rules for {filename}:\n{file_rules}\n")

    merged_rules = dict()
    for filename, file_rules in rules.items():
        merged_rules[filename] = qca.merge_rules(file_rules, filename)

        logger.info(f"merged rules for {filename}:\n{merged_rules[filename]}\n")

    # rules = rules[sets[0].attrs["file"]]
    # merged_rules = qca.merge_rules(rules)
    # merged_rules = merged_rules[sets[0].attrs["file"]]

    for filename, file_rules in merged_rules.items():
        necessary_rules = qca.get_minimal_necessary_rules(file_rules, filename)

        logger.info(f"necessary rules for {filename}:\n{necessary_rules}\n")
