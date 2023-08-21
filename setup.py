from setuptools import setup


setup(
    name="papygreektagger",
    version="0.1",
    url="https://github.com/erikhenriksson/papygreek-tagger",
    author="Erik Henriksson",
    author_email="erik.ilmari.henriksson@gmail.com",
    description="A postagger for PapyGreek texts",
    packages=[
        "papygreektagger",
        "papygreektagger.tagger",
        "papygreektagger.lm.SuperPeitho-v1",
    ],
    package_data={"": ["*.txt", "*.json", "*.pt", "*.bin"]},
    install_requires=["flair", "regex"],
    zip_safe=False,
)
