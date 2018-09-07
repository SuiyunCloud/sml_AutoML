class Base(object):
    def __init__(self,prop):
        self.prop = prop
        self.prop2 = prop

class speedml(Base):
    def __init__(self, train):
        Base.train = train

    def ptrain(self):
        print("Base.train is " + str(Base.train))
        print("self.train is " + str(self.train))

sml = speedml(1)
sml.train
sml.ptrain()
sml.train = 2
sml.train
sml.ptrain()
