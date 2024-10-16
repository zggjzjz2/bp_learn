# 定义一个类 BankAccount
class BankAccount:
    def __init__(self, owner, balance=0):
        # __init__ 是构造函数，定义初始化时需要传入的参数
        self.owner = owner  # 定义账户持有人
        self.balance = balance  # 定义账户余额，默认为 0

    def deposit(self, amount):
        # 存款方法，增加账户余额
        self.balance += amount
        print(f"存入 {amount} 元。当前余额为 {self.balance} 元。")

    def withdraw(self, amount):
        # 取款方法，减少账户余额
        if amount > self.balance:
            print(f"余额不足。当前余额为 {self.balance} 元。")
        else:
            self.balance -= amount
            print(f"取出 {amount} 元。当前余额为 {self.balance} 元。")


# 创建账户实例
my_account = BankAccount("Alice", 100)  # 持有人是 Alice，初始余额 100 元
my_account.deposit(50)  # 存入 50 元
my_account.withdraw(30)  # 取出 30 元
my_account.withdraw(200)  # 尝试取出超过余额的钱
