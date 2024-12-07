from fractal_tasks_core.dev.create_manifest import create_manifest
from operetta_compose import PACKAGE

if __name__ == "__main__":
    AUTHORS="Fabio Steffen"
    create_manifest(
        package=PACKAGE,
        authors=AUTHORS,
        docs_link="https://leukemia-kispi.github.io/operetta-compose/",
        custom_pydantic_models=[
            ("operetta_compose.io", "__init__.py", "OmeroNgffWindow"),
            ("operetta_compose.io", "__init__.py", "OmeroNgffChannel"),
        ],
    )
