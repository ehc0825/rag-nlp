# rag-nlp

#### notice

- pynndescent/distances.py 의 해당 function 을 수정하여 사용

#### original code
```
@numba.vectorize(fastmath=True)
def correct_alternative_cosine(d):
    return 1.0 - pow(2.0, -d)
```

#### change code
```
@numba.njit(fastmath=True)
def correct_alternative_cosine(ds):
    result = np.empty_like(ds)
    for i in range(ds.shape[0]):
        result[i] = 1.0 - np.power(2.0, ds[i])
    return result
```