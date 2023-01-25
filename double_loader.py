class DoubleLoader:
    
    def __init__(self, dataloader_A, dataloader_B) -> None:
        self.dataloader_A = dataloader_A
        self.dataloader_B = dataloader_B
        
        self.dataloader_A_iter = None
        self.dataloader_B_iter = None
        
        self.i = 0

    def __iter__(self):
        self.i = 0
        self.dataloader_A_iter = self.dataloader_A.__iter__()
        self.dataloader_B_iter = self.dataloader_B.__iter__()
        return self
    
    def __next__(self):
        if self.i % 2 == 0:
            nxt = self.dataloader_A_iter.__next__()
            
        else:
            nxt = self.dataloader_B_iter.__next__()
        
        self.i += 1
        return nxt

    def __len__(self):
        return min(self.dataloader_A.__len__(), self.dataloader_B.__len__())

# Unit Test
class DoubleLoaderTest:
    def __init__(self,data=1):
        self.old_data = data

    def __iter__(self):
        self.data = self.old_data
        return self

    def __next__(self):
        if self.data > 5:
            raise StopIteration
        else:
            self.data+=1
            return self.data

if __name__ == "__main__":
    loader_A = DoubleLoaderTest(3)
    loader_B = DoubleLoaderTest(2)
    
    loader = DoubleLoader(loader_A, loader_B)
    
    for item in loader:
        print(item)
    print("Try Load Again")
    for item in loader:
        print(item)
