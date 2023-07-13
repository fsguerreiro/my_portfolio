import os
import shutil


class OrganizadorArquivos():
    def __init__(self, pastas, pasta_alvo):
        self.pastas = pastas
        self.pasta_alvo = pasta_alvo


    def criar_pastas(self):
        [os.mkdir(os.path.join(self.pasta_alvo, fd)) for fd in self.pastas.keys()
         if not os.path.exists(os.path.join(self.pasta_alvo, fd))]


    def mover_arquivos(self):
        for item in os.listdir(self.pasta_alvo):
            if not os.path.isdir(os.path.join(self.pasta_alvo, item)):
                _, ext = os.path.splitext(item)
                for folder, extensions in self.pastas.items():
                    if ext.lower() in extensions:
                        [shutil.move(os.path.join(self.pasta_alvo, item),os.path.join(self.pasta_alvo, folder))]
                        print(f"Arquivo '{item}' movido para a pasta '{folder}'")
                        break

                    
    def apagar_pasta_vazia(self):
        for root, dirs, files in os.walk(self.pasta_alvo, topdown=False):
            for directory in dirs:
                directory_path = os.path.join(root, directory)
                if not os.listdir(directory_path):  # Check if the directory is empty
                    try:
                        os.rmdir(directory_path)
                        print(f"Pasta '{directory_path}' deletada!")
                    except OSError as e:
                        print(f"Falha em apagar pasta: {directory_path}, Erro: {str(e)}")
    
    
pasta_alvo = r'C:\Users\ferna\Desktop\Python_files'

pastas = {'Arquivos de figura': ['.png','.bmp','.jpg','.jpeg'],
          'Arquivos de python': ['.py','.ipynb'],
          'Arquivos de dados': ['.data','.names','.txt','.csv','.xlsx','.xls'],
          'Arquivos de doc e pdf': ['.pdf','.doc','.docx'],
          'Arquivos executáveis e zipados': ['.exe', '.zip', '.rar']
         }

organizador = OrganizadorArquivos(pastas, pasta_alvo)
organizador.criar_pastas()
organizador.mover_arquivos()
organizador.apagar_pasta_vazia()




        

