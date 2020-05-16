
class Father(object):
    def __init__(self, name):
        self.name = name
        print("Father ___init__: %s" % self.name)

    def getName(self):
        return 'Father ' + self.name


class Son(Father):
    def __init__(self, name):
        super(Son, self).__init__(name)
        print("Son __init__: %s" % self.name)
        self.name =  name

    def getName(self):

        return 'Son '+self.name + super().getName()


if __name__=='__main__':
    son=Son('yang')
    print(son.getName())