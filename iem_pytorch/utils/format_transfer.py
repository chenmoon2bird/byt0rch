import json


def to_np(x):
    return x.detach().cpu().numpy()


def to_float(xs):
    return [float(x) for x in xs]


def to_json(xs):
    return [json.dumps(to_float(x)) for x in xs]
