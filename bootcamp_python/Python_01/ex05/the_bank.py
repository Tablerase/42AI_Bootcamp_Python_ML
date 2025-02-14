import sys
from typing import List, Dict


class Account(object):
    ID_COUNT = 1

    def __init__(self, name, **kwargs):
        self.__dict__.update(kwargs)
        self.id = self.ID_COUNT
        Account.ID_COUNT += 1
        self.name = name
        if not hasattr(self, 'value'):
            self.value = 0
        if self.value < 0:
            raise AttributeError("Attribute value cannot be negative.")
        if not isinstance(self.name, str):
            raise AttributeError("Attribute name must be a str object.")

    def transfer(self, amount):
        self.value += amount


class Bank(object):
    """The bank"""

    def __init__(self):
        self.accounts = []

    def check_account(new_account: Account):
        """Check if the account is corrupted

        Args:
            new_account (Account): account to check
        """
        # right object
        if not isinstance(new_account, Account):
            raise AttributeError(
                f"Invalid object only Account supported, got {type(new_account)}")
        # odd num of attr
        attr_items = vars(new_account).items()
        amount_of_attr = len(attr_items)
        if not amount_of_attr % 2 == 1:
            raise AttributeError(
                f"Attribute amount not odd: {amount_of_attr}")
        # checking attr
        # attr starting with b
        start_with_b = [attr for attr,
                        values in attr_items if attr.startswith("b")]
        if len(start_with_b) > 0:
            raise AttributeError(
                f"Attribute cant start with b: {start_with_b}")
        # no zip or addr start
        start_with_zip = [attr for attr,
                          values in attr_items if attr.startswith("zip")]
        start_with_addr = [attr for attr,
                           values in attr_items if attr.startswith("addr")]
        if not (len(start_with_addr) >= 1 or len(start_with_zip) >= 1):
            raise AttributeError(
                f"Account should have attributes starting with zip and addr: {attr_items}")
        # no name, id, value
        if (not hasattr(new_account, "name") or
                not hasattr(new_account, "id") or
                not hasattr(new_account, "value")):
            raise AttributeError(
                f"Account should have attributes name, id, value: {attr_items}")
        # name isnt str
        if not isinstance(new_account.name, str):
            raise AttributeError(
                f"Account name should be a str: {new_account.name}")
        # id isnt int
        if not isinstance(new_account.id, int):
            raise AttributeError(
                f"Account id should be an int: {new_account.id}"
            )
        # value isnt int or float
        if not isinstance(new_account.value, (int, float)):
            raise AttributeError(
                f"Account value should be an int or float: {new_account.value}"
            )

    def add(self, new_account):
        """ Add new_account in the Bank
            @new_account: Account() new account to append
            @return True if success, False if an error occured
        """
        try:
            # test if new_account is an Account() instance and if
            # it can be appended to the attribute accounts
            Bank.check_account(new_account)
            for account in self.accounts:
                if account.name == new_account.name:
                    return False
            self.accounts.append(new_account)
            return True
        except AttributeError:
            return False

    def transfer(self, origin, dest, amount):
        """" Perform the fund transfer
            @origin: str(name) of the first account
            @dest: str(name) of the destination account
            @amount: float(amount) amount to transfer
            @return True if success, False if an error occured
        """
        source_account: Account = next(
            (x for x in self.accounts if x.name == origin), None)
        dest_account: Account = next((
            x for x in self.accounts if x.name == dest), None)
        try:
            Bank.check_account(source_account)
            Bank.check_account(dest_account)
        except:
            return False
        if source_account is None or dest_account is None:
            return False
        # print(source_account.__dict__, dest_account.__dict__)
        if not amount > 0:
            return False
        if amount >= source_account.value:
            return False
        dest_account.value += amount
        source_account.value -= amount
        return True

    def fix_account(self, name):
        """ fix account associated to name if corrupted
            @name: str(name) of the account
            @return True if success, False if an error occured
        """
        # ... No account to fix because u cant add an account without check
        # if u overide bank.accounts.append with a check_account
        print("....Fixing account...")
        account = next((x for x in self.accounts if x.name == name), None)
        if account is None:
            return False

        # Remove corrupt attributes (starting with 'b')
        for attr, value in list(vars(account).items()):
            if attr.startswith('b'):
                new_attr = attr[1:]
                delattr(account, attr)
                setattr(account, new_attr, value)

        # Add missing required attributes
        required_attrs = ['name', 'id', 'value']
        for attr in required_attrs:
            if not hasattr(account, attr):
                setattr(account, attr, None)

        # Add addr or zip if missing
        if not any(attr.startswith(('addr', 'zip')) for attr in vars(account)):
            setattr(account, 'addr', 'unknown')

        return True


if __name__ == "__main__":
    print(f"{'':_^60}")
    test = Account(
        'William John',
        zip='100-064',
        value=6460.0,
        ref='58ba2b9954cd278eda8a84147ca73c87',
        info=None,
        other='This is the vice president of the corporation'
    )
    Bank.check_account(test)

    print(f"{'Bank add':_^60}")

    bank = Bank()
    print(bank.add(test))

    print(f"{'Account Corrupted':_^60}")

    test_corrupted = Account(
        'Smith Jane',
        zip='911-745',
        value=1000.0,
        bref='1044618427ff2782f0bbece0abd05f31'
    )
    print(bank.add(test_corrupted))

    print(f"{'Transfert Invalid amount':_^60}")
    bank.add(Account(
        'Smith Jane',
        zip='911-745',
        value=1000.0,
        ref='1044618427ff2782f0bbece0abd05f31'
    ))
    print("accounts:", bank.accounts)
    if bank.transfer('William John', 'Smith Jane', 8000.0) is False:
        print('Failed')

    print(f"{'Adding corrupted and fixing it':_^60}")
    bank.accounts.append(Account(
        'Corrup Ted',
        zip='911-745',
        value=1000.0,
        bref='1044618427ff2782f0bbece0abd05f31'
    ))

    if bank.transfer('William John', 'Corrup Ted', 1000.0) is False:
        print('Failed')
        bank.fix_account('William John')
        bank.fix_account('Corrup Ted')

    print(f"{'Transfert after fix':_^60}")

    if bank.transfer('William John', 'Corrup Ted', 1000.0) is False:
        print('Failed')
    else:
        print('Success')
