from flask import Flask, request, jsonify, render_template, send_file
import logging
from model_service import ModelService
from transaction_storage import TransactionStorage
import csv
import io
from datetime import datetime

app = Flask(__name__)

# Basic startup log
logging.basicConfig(level=logging.INFO)
app.logger.info("Fraud Detection Flask app initialized.")

# Initialize service
try:
    service = ModelService.get_instance()
    app.logger.info("Model service initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize model service: {e}")
    service = None

# Initialize transaction storage
try:
    storage = TransactionStorage()
    app.logger.info("Transaction storage initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize transaction storage: {e}")
    storage = None

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Template error: {str(e)}")
        return f"Error loading template: {str(e)}", 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not service:
        return jsonify({'error': 'Model service not available'}), 500
    
    try:
        data = request.get_json(silent=True) or {}
        transaction_id = data.get('transaction_id', '')
        time_value = data.get('time', 'now')
        amount_value = data.get('amount', 0)
        result = service.predict(transaction_id, time_value, amount_value)
        
        # Store transaction if no error
        if storage and 'error' not in result:
            storage.save_transaction(result)
        
        return jsonify(result)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({
        'status': 'ok',
        'model_loaded': service is not None,
        'storage_available': storage is not None,
        'message': 'Service healthy'
    })

@app.route('/api/transactions', methods=['GET'])
def api_get_transactions():
    """Get transaction history with pagination"""
    if not storage:
        return jsonify({'error': 'Transaction storage not available'}), 500
    
    try:
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        transactions = storage.get_all_transactions(limit=limit, offset=offset)
        return jsonify({
            'transactions': transactions,
            'count': len(transactions)
        })
    except Exception as e:
        app.logger.error(f"Error fetching transactions: {str(e)}")
        return jsonify({'error': f'Failed to fetch transactions: {str(e)}'}), 500

@app.route('/api/transactions/<transaction_id>', methods=['GET'])
def api_get_transaction(transaction_id):
    """Get a specific transaction by ID"""
    if not storage:
        return jsonify({'error': 'Transaction storage not available'}), 500
    
    try:
        transaction = storage.get_transaction_by_id(transaction_id)
        if transaction:
            return jsonify(transaction)
        else:
            return jsonify({'error': 'Transaction not found'}), 404
    except Exception as e:
        app.logger.error(f"Error fetching transaction: {str(e)}")
        return jsonify({'error': f'Failed to fetch transaction: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def api_statistics():
    """Get aggregated statistics"""
    if not storage:
        return jsonify({'error': 'Transaction storage not available'}), 500
    
    try:
        stats = storage.get_statistics()
        return jsonify(stats)
    except Exception as e:
        app.logger.error(f"Error fetching statistics: {str(e)}")
        return jsonify({'error': f'Failed to fetch statistics: {str(e)}'}), 500

@app.route('/api/batch-predict', methods=['POST'])
def api_batch_predict():
    """Process multiple transactions at once"""
    if not service:
        return jsonify({'error': 'Model service not available'}), 500
    
    try:
        data = request.get_json(silent=True) or {}
        transactions = data.get('transactions', [])
        
        if not isinstance(transactions, list):
            return jsonify({'error': 'transactions must be a list'}), 400
        
        if len(transactions) > 100:
            return jsonify({'error': 'Maximum 100 transactions per batch'}), 400
        
        results = []
        for tx in transactions:
            transaction_id = tx.get('transaction_id', '')
            time_value = tx.get('time', 'now')
            amount_value = tx.get('amount', 0)
            
            result = service.predict(transaction_id, time_value, amount_value)
            
            # Store transaction if no error
            if storage and 'error' not in result:
                storage.save_transaction(result)
            
            results.append(result)
        
        return jsonify({
            'results': results,
            'count': len(results),
            'processed': len([r for r in results if 'error' not in r])
        })
    except Exception as e:
        app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/api/export', methods=['GET'])
def api_export():
    """Export transactions as CSV"""
    if not storage:
        return jsonify({'error': 'Transaction storage not available'}), 500
    
    try:
        export_format = request.args.get('format', 'csv').lower()
        
        if export_format == 'csv':
            transactions = storage.get_all_transactions()
            
            if not transactions:
                return jsonify({'error': 'No transactions to export'}), 404
            
            # Create CSV in memory
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=[
                'id', 'transaction_id', 'timestamp', 'amount', 
                'fraud_prediction', 'fraud_probability', 
                'risk_level', 'confidence', 'stored_at'
            ])
            writer.writeheader()
            
            for tx in transactions:
                writer.writerow({
                    'id': tx.get('id', ''),
                    'transaction_id': tx.get('transaction_id', ''),
                    'timestamp': tx.get('timestamp', ''),
                    'amount': tx.get('amount', ''),
                    'fraud_prediction': tx.get('fraud_prediction', ''),
                    'fraud_probability': tx.get('fraud_probability', ''),
                    'risk_level': tx.get('risk_level', ''),
                    'confidence': tx.get('confidence', ''),
                    'stored_at': tx.get('stored_at', '')
                })
            
            output.seek(0)
            
            # Create response
            mem = io.BytesIO()
            mem.write(output.getvalue().encode('utf-8'))
            mem.seek(0)
            
            filename = f'fraud_detection_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            
            return send_file(
                mem,
                mimetype='text/csv',
                as_attachment=True,
                download_name=filename
            )
        
        elif export_format == 'json':
            transactions = storage.get_all_transactions()
            return jsonify({
                'export_date': datetime.now().isoformat(),
                'count': len(transactions),
                'transactions': transactions
            })
        
        else:
            return jsonify({'error': 'Invalid format. Use csv or json'}), 400
            
    except Exception as e:
        app.logger.error(f"Export error: {str(e)}")
        return jsonify({'error': f'Export failed: {str(e)}'}), 500

@app.route('/api/clear-history', methods=['POST'])
def api_clear_history():
    """Clear all transaction history (admin function)"""
    if not storage:
        return jsonify({'error': 'Transaction storage not available'}), 500
    
    try:
        success = storage.clear_history()
        if success:
            return jsonify({'message': 'Transaction history cleared successfully'})
        else:
            return jsonify({'error': 'Failed to clear history'}), 500
    except Exception as e:
        app.logger.error(f"Clear history error: {str(e)}")
        return jsonify({'error': f'Failed to clear history: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)