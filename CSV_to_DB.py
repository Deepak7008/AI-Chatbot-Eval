import json
import csv
import os

def generate_mock_db():
    users = []
    with open('Mock_csv/users.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            users.append(row)
            
    products = []
    with open('Mock_csv/products.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['price'] = float(row['price'])
            row['warranty_months'] = int(row['warranty_months'])
            row['stock_quantity'] = int(row['stock_quantity'])
            products.append(row)
            
    orders_dict = {}
    with open('Mock_csv/orders.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            oid = row['order_id']
            if oid not in orders_dict:
                order_obj = {
                    "order_id": oid,
                    "user_id": row["user_id"],
                    "status": row["status"],
                    "channel": row["channel"],
                    "items": [],
                    "order_date": row["order_date"] if row["order_date"] else None,
                    "estimated_delivery": row["estimated_delivery"] if row["estimated_delivery"] else None,
                    "actual_delivered_date": row["actual_delivered_date"] if row["actual_delivered_date"] else None,
                    "tracking_number": row["tracking_number"] if row["tracking_number"] else None,
                    "shipping_address": row["shipping_address"] if row["shipping_address"] else None
                }
                
                if row.get("cancelled_date"):
                    order_obj["cancelled_date"] = row["cancelled_date"]
                if row.get("cancellation_reason"):
                    order_obj["cancellation_reason"] = row["cancellation_reason"]
                    
                if row.get("return_initiated_date"):
                    order_obj["return_initiated_date"] = row["return_initiated_date"]
                if row.get("return_reason"):
                    order_obj["return_reason"] = row["return_reason"]
                if row.get("return_status"):
                    order_obj["return_status"] = row["return_status"]
                if row.get("refund_amount"):
                    order_obj["refund_amount"] = float(row["refund_amount"])
                if row.get("refund_date"):
                    order_obj["refund_date"] = row["refund_date"]
                if row.get("refund_status"):
                    order_obj["refund_status"] = row["refund_status"]
                
                orders_dict[oid] = order_obj
                
            # Append item
            item = {
                "product_id": row["product_id"],
                "quantity": int(row["quantity"]),
                "unit_price_at_purchase": float(row["unit_price_at_purchase"])
            }
            orders_dict[oid]["items"].append(item)
            
    mock_db = {
        "users": users,
        "products": products,
        "orders": list(orders_dict.values())
    }
    
    os.makedirs('data', exist_ok=True)
    with open('data/mock_db.json', 'w', encoding='utf-8') as f:
        json.dump(mock_db, f, indent=2)

def generate_policies():
    policies = {}
    
    with open('Mock_csv/policies.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cat = row['category']
            subcat = row['subcategory']
            key = row['grouping_key']
            val = row['value']
            
            if cat not in policies:
                policies[cat] = {}
                
            try:
                if '.' in val:
                    val_parsed = float(val)
                else:
                    val_parsed = int(val)
            except ValueError:
                val_parsed = val
                
            if subcat:
                if not key:
                    if subcat not in policies[cat]:
                        policies[cat][subcat] = []
                    policies[cat][subcat].append(val_parsed)
                else:
                    if subcat not in policies[cat]:
                        policies[cat][subcat] = {}
                    policies[cat][subcat][key] = val_parsed
            else:
                policies[cat][key] = val_parsed
                
    with open('Mock_csv/faq.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            topic = row['topic']
            key = row['question']
            val = row['answer']
            if topic not in policies:
                policies[topic] = {}
            policies[topic][key] = val
            
    with open('data/policies.json', 'w', encoding='utf-8') as f:
        json.dump(policies, f, indent=2)

def generate_single_turn():
    cases = []
    with open('Mock_csv/test_cases_single.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case = {
                "id": row["id"],
                "category": row["category"],
                "query": row["query"],
                "context": {"order_id": row["order_id"] if row["order_id"] else None},
                "reference_answer": row["reference_answer"],
                "tags": row["tags"].split(","),
                "difficulty": row["difficulty"]
            }
            cases.append(case)
            
    with open('data/dataset_single.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2)

def generate_multi_turn():
    cases_dict = {}
    
    with open('Mock_csv/test_cases_multi.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["id"]
            if cid not in cases_dict:
                cases_dict[cid] = {
                    "id": cid,
                    "category": "",
                    "turns": [],
                    "reference_answers": [],
                    "task_completed": True,
                    "expected_outcome": "",
                    "tags": []
                }
            
            # Greedily grab metadata if present on any row for this ID
            if row.get("category"): cases_dict[cid]["category"] = row["category"]
            if row.get("expected_outcome"): cases_dict[cid]["expected_outcome"] = row["expected_outcome"]
            if row.get("tags"): cases_dict[cid]["tags"] = row["tags"].split(",")
            if row.get("task_completed"): cases_dict[cid]["task_completed"] = row["task_completed"].lower() == "true"
            
            cases_dict[cid]["turns"].append({"role": "user", "content": row["user_message"]})
            cases_dict[cid]["turns"].append({"role": "assistant", "content": None})
            cases_dict[cid]["reference_answers"].append(row["reference_answer"])

    cases = list(cases_dict.values())
    
    with open('data/dataset_multi.json', 'w', encoding='utf-8') as f:
        json.dump(cases, f, indent=2)

    # Note: We intentionally do NOT reset dataset_extended.json here 
    # to protect organically gathered test cases.

if __name__ == "__main__":
    generate_mock_db()
    generate_policies()
    generate_single_turn()
    generate_multi_turn()
    print("Generated all mobile-focused datasets successfully from CSV sources.")
