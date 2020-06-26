from flowproposal.utils import replace_in_list


def test_replace_in_list():
    """
    Test if the list produced contains the correct entries in the correct
    locations
    """
    l = [1, 2, 3]
    replace_in_list(l, [1, 2], [5, 4])
    assert l == [5, 4, 3]


def test_replace_in_list_item():
    """
    Test if items are correctly converted to lists in replace_in_list function
    """
    l = [1, 2, 3]
    replace_in_list(l, 3, 4)
    assert l == [1, 2, 4]

