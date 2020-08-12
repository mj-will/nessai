from flowproposal.utils import replace_in_list


def test_replace_in_list():
    """
    Test if the list produced contains the correct entries in the correct
    locations
    """
    x = [1, 2, 3]
    replace_in_list(x, [1, 2], [5, 4])
    assert x == [5, 4, 3]


def test_replace_in_list_item():
    """
    Test if items are correctly converted to lists in replace_in_list function
    """
    x = [1, 2, 3]
    replace_in_list(x, 3, 4)
    assert x == [1, 2, 4]
