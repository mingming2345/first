# input("提示用户需要输入什么内容")
# #例如
# input("请输入您的用户名")
# #想要把你的用户名保存为变量，就新建变量存储，但input存储的是字符串类型的数据
# username=input("请输入用户名：\n")#新建变量后，你就可以运用变量来做你的需求
# #例如
# print("欢迎回家",username,'你终于回来了',sep=",")
# print("欢迎回家",int(username),type(username),type(int(username)),'你终于回来了',sep=",")
# num1=input("第一个数字:\n")
# num2=input("第二个数字\n")
# result=num1+num2
# print(result)
# int(result)
# print(type(result))
#获取用户信息，并将用户信息输出
# username=input("用户名输入:\n")
# passport=input("密码输入:\n")
# age=input(("年龄输入:\n"))
# sex=input("性别输入:\n")
# print(username,passport,age,sex,sep="\n")\
#求长方形面积
length=int(input("输入所求长方形的长:\n"))
width=int(input("所求长方形的宽:\n"))
area=length*width
print(area)
