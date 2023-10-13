import json


class Response:
    def __init__(self, result=True, code=0, message='', data={}):
        self.result = result
        self.code = code
        self.data = data
        self.message = message

    def __iter__(self):
        yield from {
            'result': self.result,
            'code': self.code,
            'data': self.data,
            'message': self.message
        }.items()

    def __str__(self):
        return json.dumps(dict(self), ensure_ascii=False)

    def __repr__(self):
        return self.__str__()
