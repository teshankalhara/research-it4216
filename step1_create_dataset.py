"""
STEP 1 ENHANCED: Create LARGE-SCALE PDF Dataset (1000+ PDFs)
NO PERSONAL INFORMATION - All data is synthetic/random
"""

import os
import pandas as pd
import random
from datetime import datetime, timedelta
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import numpy as np

# Create directories
def create_folders():
    """Create dataset folder structure"""
    folders = [
        "dataset/genuine",
        "dataset/tampered/font_change",
        "dataset/tampered/character_replace",
        "dataset/tampered/spacing_alter",
        "dataset/tampered/embedding_mismatch",
        "dataset/ground_truth"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✓ Folders created")

# Generate random names and data (NO REAL NAMES)
FIRST_NAMES = ['UserA', 'UserB', 'UserC', 'UserD', 'UserE', 'UserF', 'UserG', 'UserH', 'UserI', 'UserJ']
LAST_NAMES = ['Client01', 'Client02', 'Client03', 'Client04', 'Client05', 'Client06', 'Client07', 'Client08', 'Client09', 'Client10']
BANKS = ['Sample Bank A', 'Sample Bank B', 'Sample Bank C', 'Sample Bank D', 'Sample Bank E']
CITIES = ['City01', 'City02', 'City03', 'City04', 'City05', 'City06', 'City07', 'City08']
VEHICLES = ['VehicleMakeA', 'VehicleMakeB', 'VehicleMakeC', 'VehicleMakeD', 'VehicleMakeE']
CERTIFICATES = ['Bachelor of Sample', 'Master of Sample', 'Diploma in Sample', 'PhD Sample', 'Certificate in Sample']

def generate_random_amount(min_val=100, max_val=100000):
    """Generate random amount"""
    return round(random.uniform(min_val, max_val), 2)

def generate_random_date(start_date='2020-01-01', end_date='2025-12-31'):
    """Generate random date"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)

def generate_bank_statement(doc_id):
    """Generate random bank statement content"""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    bank = random.choice(BANKS)
    beginning_balance = generate_random_amount(1000, 50000)
    deposits = generate_random_amount(500, 10000)
    withdrawals = generate_random_amount(200, 5000)
    ending_balance = beginning_balance + deposits - withdrawals
    
    transactions = []
    for i in range(random.randint(3, 8)):
        trans_date = generate_random_date('2024-01-01', '2024-12-31')
        trans_type = random.choice(['Deposit', 'Withdrawal', 'Transfer', 'Interest'])
        trans_amount = generate_random_amount(50, 5000)
        transactions.append(f"{trans_date.strftime('%m/%d/%Y')} - {trans_type} - ${trans_amount:,.2f}")
    
    return [
        f"{bank}",
        f"MONTHLY STATEMENT - {random.choice(['January', 'February', 'March', 'April', 'May', 'June'])} 2024",
        "-" * 50,
        f"Account Holder: {first_name} {last_name}",
        f"Account Number: ****{random.randint(1000, 9999)}",
        "",
        f"BEGINNING BALANCE: ${beginning_balance:,.2f}",
        f"DEPOSITS: ${deposits:,.2f}",
        f"WITHDRAWALS: ${withdrawals:,.2f}",
        f"ENDING BALANCE: ${ending_balance:,.2f}",
        "",
        "TRANSACTION HISTORY:",
    ] + transactions + [
        "",
        f"TOTAL DUE: ${generate_random_amount(0, 5000):,.2f}"
    ]

def generate_vehicle_registration(doc_id):
    """Generate random vehicle registration"""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    vehicle = random.choice(VEHICLES)
    city = random.choice(CITIES)
    reg_date = generate_random_date('2020-01-01', '2024-12-31')
    valid_until = reg_date + timedelta(days=365*3)
    
    return [
        "DEPARTMENT OF MOTOR VEHICLES",
        "CERTIFICATE OF REGISTRATION",
        "-" * 50,
        f"Registration Number: {random.choice(['ABC', 'XYZ', 'DEF', 'GHI'])}-{random.randint(1000, 9999)}",
        f"License Plate: {random.choice(['CA', 'NY', 'TX'])}-{random.randint(1000, 9999)}",
        f"Vehicle Make: {vehicle}",
        f"Vehicle Model: {random.choice(['Sedan', 'SUV', 'Hatchback', 'Truck'])}",
        f"Year: {random.randint(2018, 2024)}",
        f"Color: {random.choice(['Black', 'White', 'Silver', 'Blue', 'Red'])}",
        f"Owner Name: {first_name} {last_name}",
        f"Owner Address: {random.randint(1, 999)} {random.choice(['Main', 'Park', 'Lake'])} Street, {city}",
        f"Registration Date: {reg_date.strftime('%Y-%m-%d')}",
        f"Valid Until: {valid_until.strftime('%Y-%m-%d')}",
        f"Engine Number: ENG-{random.randint(100000, 999999)}",
        f"Chassis Number: CHS-{random.randint(100000, 999999)}"
    ]

def generate_academic_certificate(doc_id):
    """Generate random academic certificate"""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    certificate = random.choice(CERTIFICATES)
    grad_date = generate_random_date('2020-01-01', '2025-12-31')
    grades = random.choice(['First Class', 'Second Class Upper', 'Second Class Lower', 'Merit', 'Distinction'])
    
    return [
        f"UNIVERSITY OF {random.choice(['SAMPLE_A', 'SAMPLE_B', 'SAMPLE_C', 'SAMPLE_D'])}",
        "FACULTY OF APPLIED SCIENCE",
        "-" * 50,
        "CERTIFICATE OF DEGREE",
        "",
        "This is to certify that",
        f"{first_name} {last_name}",
        f"Registration Number: {random.randint(2010, 2024)}/SAMPLE/{random.randint(1, 999)}",
        "",
        "has successfully completed the requirements for",
        f"{certificate} in {random.choice(['Information Technology', 'Computer Science', 'Data Science', 'Cybersecurity'])}",
        "",
        "with",
        f"{grades}",
        "",
        f"Date: {grad_date.strftime('%B %d, %Y')}",
        f"Chancellor: Dr. {random.choice(['SampleX', 'SampleY', 'SampleZ'])}"
    ]

def generate_invoice(doc_id):
    """Generate random invoice"""
    company_names = ['Sample Solutions', 'Global Sample', 'Digital Sample', 'Innovation Sample', 'Smart Sample']
    customer_names = ['Sample Corp', 'Sample Ltd', 'Sample Industries', 'Sample Enterprises', 'Sample Group']
    
    company = random.choice(company_names)
    customer = random.choice(customer_names)
    inv_date = generate_random_date('2024-01-01', '2024-12-31')
    
    items = []
    subtotal = 0
    for i in range(random.randint(2, 6)):
        item_name = random.choice(['Item A', 'Item B', 'Item C', 'Item D', 'Item E'])
        qty = random.randint(1, 10)
        price = generate_random_amount(10, 2000)
        total = qty * price
        subtotal += total
        items.append(f"{item_name:<20} {qty:<8} ${price:>8,.2f}  ${total:>10,.2f}")
    
    tax = subtotal * 0.10
    total_due = subtotal + tax
    
    return [
        f"{company} (PVT) LTD",
        "TAX INVOICE",
        "-" * 50,
        f"Invoice #: INV-{random.randint(10000, 99999)}",
        f"Date: {inv_date.strftime('%Y-%m-%d')}",
        f"Customer ID: CUST-{random.randint(1000, 9999)}",
        "",
        f"Bill To: {customer}",
        f"Address: {random.randint(1, 999)} Sample Business Park, {random.choice(CITIES)}",
        "",
        f"{'ITEM':<20} {'QTY':<8} {'PRICE':<10} {'TOTAL':<12}",
        "-" * 50,
    ] + items + [
        "-" * 50,
        f"{'SUBTOTAL:':<20} {'':<8} {'':<10} ${subtotal:>10,.2f}",
        f"{'TAX (10%):':<20} {'':<8} {'':<10} ${tax:>10,.2f}",
        f"{'TOTAL DUE:':<20} {'':<8} {'':<10} ${total_due:>10,.2f}",
        "",
        f"Payment Due: {random.randint(15, 60)} days"
    ]

def generate_legal_contract(doc_id):
    """Generate random legal contract"""
    party_a = random.choice(['Sample Solutions Ltd', 'Sample Services Inc', 'Sample Innovations Corp', 'Sample Systems LLC'])
    party_b = random.choice(['Sample Client Corp', 'Sample Business Partners Ltd', 'Sample Enterprise Solutions', 'Sample Corporate Services'])
    contract_date = generate_random_date('2024-01-01', '2024-12-31')
    contract_value = generate_random_amount(10000, 500000)
    duration = random.randint(6, 36)
    
    return [
        "SERVICE AGREEMENT",
        "-" * 50,
        f"Agreement #: AG-{random.randint(10000, 99999)}",
        f"Date: {contract_date.strftime('%Y-%m-%d')}",
        "",
        "Between:",
        f"Party A: {party_a}",
        f"Party B: {party_b}",
        "",
        f"Contract Value: ${contract_value:,.2f}",
        f"Duration: {duration} months",
        f"Start Date: {(contract_date + timedelta(days=30)).strftime('%Y-%m-%d')}",
        "",
        "Terms and Conditions:",
        "1. Services shall be provided as per scope of work",
        "2. Payment terms: Net 30 days upon invoice",
        "3. Late payment interest: 1.5% per month",
        "4. Termination: 30 days written notice",
        "5. Governing law: Sample Jurisdiction",
        "",
        "Authorized Signatures:",
        "_____________________",
        f"{party_a} Representative",
        "",
        "_____________________",
        f"{party_b} Representative"
    ]

# Document type generators
DOCUMENT_GENERATORS = {
    'bank_statement': generate_bank_statement,
    'vehicle_registration': generate_vehicle_registration,
    'academic_certificate': generate_academic_certificate,
    'invoice': generate_invoice,
    'legal_contract': generate_legal_contract
}

# Tampering functions
def create_genuine_pdf(filename, content):
    """Create authentic PDF document"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 10)
    
    for line in content:
        if len(line) > 80:
            for i in range(0, len(line), 80):
                c.drawString(50, y, line[i:i+80])
                y -= 15
        else:
            c.drawString(50, y, line)
            y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def create_tampered_font_change(original_content, filename):
    """Tamper by changing font"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    
    for line in original_content:
        if any(keyword in line for keyword in ['$', 'Balance', 'Total', 'Amount', 'Value', 'Due']):
            c.setFont("Times-Bold", 11)
            c.drawString(50, y, line)
            c.setFont("Helvetica", 10)
        else:
            c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def create_tampered_char_replace(original_content, filename):
    """Tamper by replacing characters"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 10)
    
    for line in original_content:
        import re
        amounts = re.findall(r'\$[\d,]+\.?\d*', line)
        for amount in amounts:
            try:
                num = float(amount.replace('$', '').replace(',', ''))
                new_amount = f"${num * 10:,.2f}"
                line = line.replace(amount, new_amount)
            except:
                pass
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def create_tampered_spacing(original_content, filename):
    """Tamper by altering spacing"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica", 10)
    
    for line in original_content:
        if '$' in line:
            line = line.replace('$', '$ ')
            line = line.replace(',', ' , ')
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

def create_tampered_embedding(original_content, filename):
    """Tamper with font embedding"""
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    y = height - 50
    
    for i, line in enumerate(original_content):
        if i % 3 == 0:
            c.setFont("Times-Roman", 10)
        elif i % 3 == 1:
            c.setFont("Courier", 10)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(50, y, line)
        y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()

TAMPERING_FUNCTIONS = {
    'font_change': create_tampered_font_change,
    'character_replace': create_tampered_char_replace,
    'spacing_alter': create_tampered_spacing,
    'embedding_mismatch': create_tampered_embedding
}

def build_large_dataset(num_documents=200):
    """Create large-scale dataset"""
    
    print("="*60)
    print("STEP 1 ENHANCED: CREATING LARGE-SCALE PDF DATASET")
    print(f"Target: {num_documents} documents per type = {num_documents * 5} total PDFs")
    print("="*60)
    
    create_folders()
    ground_truth = []
    
    doc_types = list(DOCUMENT_GENERATORS.keys())
    
    for doc_idx in range(num_documents):
        doc_type = random.choice(doc_types)
        doc_id = f"{doc_type.upper()}_{doc_idx+1:04d}"
        
        generator = DOCUMENT_GENERATORS[doc_type]
        content = generator(doc_id)
        
        if (doc_idx + 1) % 50 == 0:
            print(f"Progress: {doc_idx + 1}/{num_documents} documents generated...")
        
        # Genuine PDF
        genuine_path = f"dataset/genuine/{doc_type}_{doc_id}_genuine.pdf"
        create_genuine_pdf(genuine_path, content)
        ground_truth.append({
            'filename': f"genuine/{doc_type}_{doc_id}_genuine.pdf",
            'label': 'genuine',
            'tampering_type': 'none',
            'doc_type': doc_type,
            'doc_id': doc_id
        })
        
        # Tampered versions
        for tamper_name, tamper_func in TAMPERING_FUNCTIONS.items():
            tampered_path = f"dataset/tampered/{tamper_name}/{doc_type}_{doc_id}_tampered.pdf"
            tamper_func(content, tampered_path)
            ground_truth.append({
                'filename': f"tampered/{tamper_name}/{doc_type}_{doc_id}_tampered.pdf",
                'label': 'tampered',
                'tampering_type': tamper_name,
                'doc_type': doc_type,
                'doc_id': doc_id
            })
    
    # Save ground truth
    df = pd.DataFrame(ground_truth)
    df.to_csv("dataset/ground_truth/labels.csv", index=False)
    
    # Statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"✓ Documents per type: {num_documents}")
    print(f"✓ Genuine PDFs: {len(df[df['label']=='genuine'])}")
    print(f"✓ Tampered PDFs: {len(df[df['label']=='tampered'])}")
    print(f"✓ TOTAL PDFs: {len(df)}")
    print(f"✓ Ground Truth: dataset/ground_truth/labels.csv")
    
    return df

if __name__ == "__main__":
    # Choose your dataset size
    # 200 = 1,000 PDFs | 500 = 2,500 PDFs | 1000 = 5,000 PDFs
    NUM_DOCUMENTS = 500  # Recommended for good research
    
    df = build_large_dataset(num_documents=NUM_DOCUMENTS)
    
    print("\n" + "="*60)
    print("✅ LARGE-SCALE DATASET CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Dataset location: dataset/")
    print(f"📊 Total PDFs: {len(df)}")
    print("\n⚠ IMPORTANT: No personal information used - All data is synthetic")
    print("\nNEXT: Run STEP 2 - Extract Font Metadata")