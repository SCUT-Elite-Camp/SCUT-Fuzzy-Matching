import numpy as np
from projection.random_hyperplane import make_planes, hash_signatures, collision_mask, projected_candidates

def test_identical_vectors_always_collide():
    vectors=np.array([[1.0,0.0],[1.0,0.0],[-1.0,0.0]])
    planes=make_planes(2,4,8,seed=6)
    hashes=hash_signatures(vectors,planes)
    mask=collision_mask(hashes[0],hashes)
    assert mask[0] and mask[1]
    assert not mask[2]

def test_round_union_and_member_restriction():
    db=np.array([[[0,0],[0,0]],[[1,1],[0,0]],[[1,1],[1,1]]],dtype=bool)
    q=np.array([[0,0],[0,0]],dtype=bool)
    assert collision_mask(q,db).tolist()==[True,True,False]
    assert projected_candidates(np.array([1,2]),q,db).tolist()==[1]

def test_invalid_dimensions():
    import pytest
    with pytest.raises(ValueError): make_planes(2,0,8,7)
    with pytest.raises(ValueError): hash_signatures(np.ones((2,4)), np.ones((3,5,2)))
