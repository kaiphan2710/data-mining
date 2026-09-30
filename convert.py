import pandas as pd

# 40 predictor attributes from census-income.names
columns = [
    "age",
    "class_of_worker",
    "detailed_industry_recode",
    "detailed_occupation_recode",
    "education",
    "wage_per_hour",
    "enroll_in_edu_inst_last_wk",
    "marital_status",
    "major_industry_code",
    "major_occupation_code",
    "race",
    "hispanic_origin",
    "sex",
    "member_of_labor_union",
    "reason_for_unemployment",
    "full_or_part_time_employment_status",
    "capital_gains",
    "capital_losses",
    "dividends_from_stocks",
    "tax_filer_status",
    "region_of_previous_residence",
    "state_of_previous_residence",
    "detailed_household_and_family_status",
    "detailed_household_summary",
    "instance_weight",
    "migration_code_change_in_msa",
    "migration_code_change_in_region",
    "migration_code_move_within_region",
    "live_in_this_house_1_year_ago",
    "migration_previous_residence_in_sunbelt",
    "num_persons_worked_for_employer",
    "family_members_under_18",
    "country_of_birth_father",
    "country_of_birth_mother",
    "country_of_birth_self",
    "citizenship",
    "own_business_or_self_employed",
    "fill_inc_questionnaire_for_veterans_admin",
    "veterans_benefits",
    "weeks_worked_in_year",
    "year",
    "income_class"
]

def load_census_file(path):
    df = pd.read_csv(
        path,
        header=None,
        names=columns,
        skipinitialspace=True,
        keep_default_na=False
    )

    # Strip accidental leading/trailing whitespace in text columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


# Load raw files
train = load_census_file("census-income.data")
test = load_census_file("census-income.test")


# Basic validation
print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain target:")
print(train["income_class"].value_counts())

print("\nTest target:")
print(test["income_class"].value_counts())

print("\nColumns:")
print(train.columns.tolist())


# Save standardized raw CSV files
train.to_csv(
    "census_income_train_raw.csv",
    index=False
)

test.to_csv(
    "census_income_test_raw.csv",
    index=False
)

print("\nSaved:")
print("census_income_train_raw.csv")
print("census_income_test_raw.csv")