# Assert environment health across different hardware & software configurations
# pytest only recognizes files that start with test!

def test_smoke():
    assert 1+1==2