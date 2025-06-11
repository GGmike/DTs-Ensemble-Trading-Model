def ck_dup_list(list_1, list_2):
  # temp1 = list_1.sort()
  # temp2 = list_2.sort()
  # return list_1 == list_2
  return set(list_1) == set(list_2)

def cal_nCr(n,r):
  temp = n - r
  return (cal_fact(n) / (cal_fact(r) * cal_fact(temp)))

def cal_fact(num):
  if num == 0 or num == 1:
    return 1
  else:
    return(num*cal_fact(num-1))