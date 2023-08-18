from setuptools import setup

setup(
    name="papygreektagger",
    version="0.1",
    url="https://github.com/erikhenriksson/papygreek-tagger",
    author="Erik Henriksson",
    author_email="erik.ilmari.henriksson@gmail.com",
    description="A postagger for PapyGreek texts",
    packages=["papygreektagger"],
    install_requires=["flair", "regex"],
    zip_safe=False,
)
