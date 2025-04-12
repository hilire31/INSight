

def concaten(l):
    ret=""
    for i in l:
        ret+=i+" "
    return ret


def filtre(string:str):
    for i in string:
        if i.isnumeric():
            return False
        if i.isupper():
            return False
    if len(string)>5:
        return False
    else: return True



def refine(string:str,filtre):
    l=string.split()
    li=[]
    for i in l:
        if not filtre(i.strip()):
            li.append(i.strip())
    return concaten(li)

i="DFGDHJ aa aaa aaaaaaa    487aaaa  Ad AA 4a"

print(refine(i,filtre))

print(filtre("aa"))