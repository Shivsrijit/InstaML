import bcrypt

hashes = {
    "Shivsrijit": "$2b$12$3h1/d0TBiTJS55lzrbyKWOaSmWzbq/NyvSCsdZ8IY8h3zAicSwmW6",
    "testuser": "$2b$12$U/hXIEDBdcUTUUbftHDeN.I3hdeI7EmyGQi.iJ2lQpCtnbfUxSSky",
    "TestUser": "$2b$12$aIJ2KHVux9os7g5o87cDzed04Nnw1z9BQJMxvKDh73K8IjcpQSaIO"
}

passwords = ["securepassword123", "password", "123456", "admin", "test", "testuser", "shivsrijit", "Shivsrijit", "pass", "12345678"]

for name, hashed in hashes.items():
    print(f"Checking {name}...")
    for p in passwords:
        try:
            if bcrypt.checkpw(p.encode('utf-8'), hashed.encode('utf-8')):
                print(f"  FOUND password for {name}: '{p}'")
                break
        except Exception as e:
            print(f"  Error checking '{p}': {e}")
