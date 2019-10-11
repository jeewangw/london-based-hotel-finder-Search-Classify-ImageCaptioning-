import self as self


class Employee:
    def __init__(self, firstName, lastName):
        self.firstName = firstName
        self.lastName = lastName

    @property
    def toJSON(self):
        return {"Employees": {'firstName': self.firstName,
                              'lastName': self.lastName}}

