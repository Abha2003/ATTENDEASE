import sqlite3

DATABASE_NAME = 'database/attendance.db'

def init_db():
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS teachers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            subject TEXT NOT NULL,
            UNIQUE(class, subject) -- Ensures a unique combination of class and subject for a teacher
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            class TEXT NOT NULL,
            roll_no TEXT NOT NULL,
            parent_email TEXT,
            parent_phone TEXT,
            address TEXT,
            dob TEXT,
            UNIQUE(roll_no, class) -- Ensures a unique combination of roll_no and class for a student
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(id),
            UNIQUE(student_id, date) -- Ensures a student can only have one attendance record per day
        )
    ''')
    conn.commit()

    # Add new columns to students table if they don't exist
    # These are for backward compatibility if the table existed before the schema update in app.py
    try:
        c.execute("ALTER TABLE students ADD COLUMN parent_email TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    try:
        c.execute("ALTER TABLE students ADD COLUMN parent_phone TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    try:
        c.execute("ALTER TABLE students ADD COLUMN address TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists
    try:
        c.execute("ALTER TABLE students ADD COLUMN dob TEXT")
    except sqlite3.OperationalError:
        pass # Column already exists

    conn.close()

def register_teacher(name, class_name, subject):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    # Convert inputs to uppercase before saving
    name = name.upper()
    class_name = class_name.upper()
    subject = subject.upper()
    try:
        c.execute("INSERT INTO teachers (name, class, subject) VALUES (?, ?, ?)", (name, class_name, subject))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # This occurs if the (class, subject) unique constraint is violated
        return False
    finally:
        conn.close()

def validate_teacher_login(class_name, subject):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    # Convert inputs to uppercase for validation
    class_name = class_name.upper()
    subject = subject.upper()
    c.execute("SELECT * FROM teachers WHERE class=? AND subject=?", (class_name, subject))
    teacher = c.fetchone()
    conn.close()
    return teacher is not None

def get_students_by_class(class_name):
    conn = sqlite3.connect(DATABASE_NAME)
    c = conn.cursor()
    # Convert input to uppercase for query
    class_name = class_name.upper()
    c.execute("SELECT id, name, class, roll_no FROM students WHERE class=?", (class_name,))
    students = c.fetchall()
    conn.close()
    # Return names and roll_numbers as uppercase for consistency, though app.py also converts
    return [(s[0], s[1].upper(), s[2].upper(), s[3].upper()) for s in students]

# Call init_db when the module is imported to ensure the database schema is up-to-date
init_db()

if __name__ == '__main__':
    # This block runs only when database.py is executed directly
    # Can be used for testing database operations
    init_db()
    print("Database initialized/updated.")

    # Example registration (will only work if not already registered)
    # if register_teacher("JOHN DOE", "GRADE 10", "MATH"):
    #     print("Teacher JOHN DOE registered for GRADE 10 MATH.")
    # else:
    #     print("Teacher JOHN DOE for GRADE 10 MATH already exists.")

    # Example login validation
    # if validate_teacher_login("GRADE 10", "MATH"):
    #     print("Login successful for GRADE 10 MATH.")
    # else:
    #     print("Login failed for GRADE 10 MATH.")
    
    # Example get students
    # students = get_students_by_class("GRADE 10")
    # print("Students in GRADE 10:", students)
