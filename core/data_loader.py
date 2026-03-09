import os
import io
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    """Handles loading and preprocessing for CMAPSS FD001 dataset."""
    
    @staticmethod
    def _read_file(f):
        if isinstance(f, (str, os.PathLike)):
            return pd.read_csv(f, sep=r"\s+", header=None)
        # Ensure we are at the start of the file in case it was pooled/read before
        if hasattr(f, 'seek'):
            f.seek(0)
        return pd.read_csv(io.BytesIO(f.read()), sep=r"\s+", header=None)

    @classmethod
    def load_from_files(cls, train_file, test_file, rul_file):
        cols = ['unit', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f'sensor_{i}' for i in range(1, 22)]

        train = cls._read_file(train_file)
        test = cls._read_file(test_file)
        rul = cls._read_file(rul_file)

        train.columns = cols
        test.columns = cols
        rul.columns = ['RUL']

        # RUL & label
        max_cycle = train.groupby('unit')['cycle'].max().reset_index(name='max_cycle')
        train = train.merge(max_cycle, on='unit')
        train['RUL'] = train['max_cycle'] - train['cycle']
        train['label'] = (train['RUL'] <= 30).astype(int)

        # Drop constants
        feat_cols = [c for c in train.columns if c not in ['unit', 'cycle', 'RUL', 'label', 'max_cycle']]
        const_cols = [c for c in feat_cols if train[c].std() == 0]
        
        if const_cols:
            train = train.drop(columns=const_cols)
            test = test.drop(columns=const_cols)
            feat_cols = [c for c in feat_cols if c not in const_cols]

        # Scale
        scaler = MinMaxScaler()
        train[feat_cols] = scaler.fit_transform(train[feat_cols])
        
        test_scaled = test.copy()
        test_scaled[feat_cols] = scaler.transform(test_scaled[feat_cols])

        # Test last cycle
        test_last = test_scaled.groupby('unit').last().reset_index()
        X_test_last_scaled = test_last[feat_cols].values
        
        # Binary labels for test set (RUL <= 30)
        y_true_all = (rul['RUL'].values <= 30).astype(int)
        
        # Align lengths in case of subsetted files or mismatched RUL uploads
        if len(y_true_all) > len(X_test_last_scaled):
            y_test_last = y_true_all[:len(X_test_last_scaled)]
        elif len(y_true_all) < len(X_test_last_scaled):
            X_test_last_scaled = X_test_last_scaled[:len(y_true_all)]
            y_test_last = y_true_all
        else:
            y_test_last = y_true_all

        return {
            "train": train,
            "test_scaled": test_scaled,
            "feat_cols": feat_cols,
            "scaler": scaler,
            "X_test_last_scaled": X_test_last_scaled,
            "y_test_last": y_test_last,
            "const_cols": const_cols
        }

    @classmethod
    def load_local_fd001(cls, base_dir="CMaps"):
        train_path = os.path.join(base_dir, 'train_FD001.txt')
        test_path = os.path.join(base_dir, 'test_FD001.txt')
        rul_path = os.path.join(base_dir, 'RUL_FD001.txt')
        return cls.load_from_files(train_path, test_path, rul_path)
