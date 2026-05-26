import numpy as np


class DEMLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.header = {}
        self.data = None

    def load(self):
        print(f"Завантаження файлу {self.filepath}...")
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

            header_count = 0
            keywords = {'ncols', 'nrows', 'xllcorner', 'yllcorner', 'xllcenter', 'yllcenter', 'cellsize',
                        'nodata_value'}

            for line in lines:
                parts = line.strip().split()
                if not parts:
                    header_count += 1
                    continue
                if parts[0].lower() in keywords:
                    self.header[parts[0].lower()] = float(parts[1])
                    header_count += 1
                else:
                    break

            data_text = " ".join(lines[header_count:])
            flat_data = np.fromstring(data_text, sep=' ')

            nrows = int(self.header['nrows'])
            ncols = int(self.header['ncols'])

            expected_size = nrows * ncols
            actual_size = len(flat_data)

            if actual_size != expected_size:
                print(f" Очікувалось {expected_size} точок, але знайдено {actual_size}.")
                nrows = actual_size // ncols
                flat_data = flat_data[:nrows * ncols]
                print(f"Коригуємо: новий розмір матриці {nrows}x{ncols}")

            self.data = flat_data.reshape((nrows, ncols))

        print(f"Успішно завантажено: {self.data.shape}")
        return self.data, self.header.get('cellsize', 1.0)  # Захист на випадок відсутності cellsize

    def get_cropped_dem(self, target_size=20):

        if self.data is None:
            self.load()

        rows, cols = self.data.shape

        if rows <= target_size and cols <= target_size:
            return self.data, self.header['cellsize']

        print(f"Вирізаємо ділянку {target_size}x{target_size} з центру...")
        start_r = rows // 2 - target_size // 2
        start_c = cols // 2 - target_size // 2

        cropped_data = self.data[start_r:start_r + target_size, start_c:start_c + target_size]

        return cropped_data, self.header['cellsize']