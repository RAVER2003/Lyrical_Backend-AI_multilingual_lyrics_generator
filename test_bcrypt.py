from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
try:
    print(pwd_context.hash("testing"))
except Exception as e:
    import traceback
    traceback.print_exc()
