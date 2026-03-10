import sys
from ..tools.execute import *

def main_print(directory, discretization_type = None, graph_type = None):
    result(directory, discretization_type, graph_type)


if __name__ == "__main__":
    args = sys.argv[1:]
    db_name = args[0]
    discretization_type = args[1]
    graph_type = args[2]

    directory = "reports/"+db_name+"/metrics/"
    main_print(directory, discretization_type, graph_type)

