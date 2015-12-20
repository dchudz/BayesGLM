# Find out what term names or column names are in a model, so you know how to specify a prior

from patsy import dmatrices


def get_term_names(formula, data):
    y, x = dmatrices(formula, data)
    return x.design_info.term_names


def get_column_names(formula, data):
    y, x = dmatrices(formula, data)
    return x.design_info.column_names
