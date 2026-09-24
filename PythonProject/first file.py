print ("hello world")
print("莎士比亚说：\"to be not or to be,that's the question\"")
print("莎士比亚说:'你是傻子'")
print("莎士比亚说：'to be not or to be,that's the question'")
print('莎士比亚说："to be not or to be,that\'s the question"')
print("2020"+"2020"+"2020"+"2020")#直接加号相连是没有空格间隔的
print("2020","2020","2020","2020")#逗号相隔它会有空格间隔
print("2020","2020","2020","2020",sep='-<')#sep参数里面填符号能让你填的符号来间隔你输出的内容
print('香蕉',end='和')#end参数默认是\n换行符，但你如果特意用end指定你想要的结尾符，它就不会自动换行
print('香蕉',end='和')
print('香蕉',end='和')
print('香蕉')
print("姓名","城市","年龄")
print("张三","李四","玄武","老六")
print('''
姓名 城市 年龄
张三 李四 玄武 老六
''')
PI = 3.1415926
fruit = "香蕉"
print("半径为1的圆的面积为：",PI*1*1)
print("半径为2的圆的面积为：",PI*2*2)
print("我最喜欢吃的水果是：",fruit)
fruit = "苹果"
print("我最喜欢吃的水果是：",fruit)
favoritefruits = "苹果"
second_fruits = favoritefruits
favoritefruits = '香蕉'
print(favoritefruits,second_fruits)