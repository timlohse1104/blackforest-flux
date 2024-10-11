from colors import bcolors

def log_session(message):
    print(f"{bcolors.OKCYAN}{message}{bcolors.ENDC}")

def log_debug(message):
    print(f"{bcolors.OKBLUE}{message}{bcolors.ENDC}")

def log_inference(message):
    print(f"{bcolors.OKGREEN}{message}{bcolors.ENDC}")

def log_error(message):
    print(f"{bcolors.FAIL}{message}{bcolors.ENDC}")

def log_warning(message):
    print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")
