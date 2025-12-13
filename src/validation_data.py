import pandera as pa

def validate_data(df, schema, raise_on_error: bool = False):
    """Validate DataFrame against schema and print results."""
    print("\n--- DATA VALIDATION ---")

    try:
        schema.validate(df, lazy=True)
        print("✓ All validation checks passed!")
        return True
    except pa.errors.SchemaErrors as err:
        print("⚠ Validation issues found:")
        print("\nFailures per column:")
        print(err.failure_cases["column"].value_counts())
        print("\nUnique errors per column:")
        print(err.failure_cases.groupby("column")["failure_case"].nunique())

        if raise_on_error:
            raise  # re-raise the same SchemaErrors

        return False
