from os import path
from common import *
from collections import defaultdict
from tqdm import tqdm

import os
import re
import glob
import subprocess as sp
import argparse
import rwb_util


def inject_prefix_rootdir(proj, bug_id):
    rpath = rwb_util.repo_path(proj, bug_id)
    return rpath
    
    
def enforce_static_assertions(gen_test):
    if 'Assert.' in gen_test:
        # force to use static assertion imports
        gen_test = gen_test.replace('Assert.fail', 'fail')
        gen_test = gen_test.replace('Assert.assert', 'assert')
    return gen_test
    

def needed_imports_by_bug_id(repo_path, proj, bug_id, gen_test):
    src_dir = rwb_util.path_prefix(proj, bug_id)

    classpaths, needed_class_stubs, needed_asserts = needed_imports(
        repo_path, src_dir, gen_test)
    
    return classpaths, needed_asserts


def inject_test_by_bug_id(repo_path, proj, bug_id, gen_test, needed_elements, dry=False):
    src_dir = rwb_util.path_prefix(proj, bug_id)
    test_dir = rwb_util.test_path_prefix(proj, bug_id)

    return inject_test(repo_path, src_dir, test_dir, gen_test, needed_elements, dry=dry)


def git_reset(repo_dir_path):
    sp.run(['git', 'reset', '--hard', 'HEAD'],
           cwd=repo_dir_path, stdout=sp.DEVNULL, stderr=sp.DEVNULL)


def git_clean(repo_dir_path):
    sp.run(['git', 'clean', '-df'],
           cwd=repo_dir_path, stdout=sp.DEVNULL, stderr=sp.DEVNULL)

    
def git_d4j_handle(repo_dir_path, ref_tag):
    sp.run(['git', 'checkout', ref_tag, '--', '.defects4j.config'],
           cwd=repo_dir_path)
    sp.run(['git', 'checkout', ref_tag, '--', 'defects4j.build.properties'],
           cwd=repo_dir_path)
    

def compile_repo(repo_dir_path):
    # actual compiling
    compile_proc = sp.run(
        ['mvn', 'compile', '-Drat.skip=true'],
        stdout=sp.PIPE, stderr=sp.PIPE, cwd=repo_dir_path)

    # extracting error message
    compile_error_lines = compile_proc.stderr.decode('utf-8').split('\n')[2:]

    compile_error_lines = [
        e for e in compile_error_lines if '[javac] [' not in e]
    compile_error_lines = [e for e in compile_error_lines if '[javac]' in e]
    compile_error_lines = [
        e for e in compile_error_lines if 'warning:' not in e]
    compile_error_lines = [
        e for e in compile_error_lines if '[javac] Note:' not in e]
    compile_error_lines = [
        e for e in compile_error_lines if 'compiler be upgraded.' not in e]
    compile_error_msg = '\n'.join(compile_error_lines)
    return compile_proc.returncode, compile_error_msg


def run_test(repo_dir_path, test_name):
    '''Returns failing test number.'''
    test_process = sp.run(['mvn', 'test', '-Drat.skip=true', f'-Dtest={test_name.split(".")[-1].replace("::", "#")}'],
                          capture_output=True, cwd=repo_dir_path)
    captured_stdout = test_process.stdout.decode()

    if len(captured_stdout) == 0:
        return -1, []  # likely compile error, all tests failed
    else:
        pattern = re.compile(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)')
        match = pattern.search(captured_stdout)
        if not match:
            return -1, [test_name.split(".")[-1].replace("::", "#")]
        failures = int(match.group(2))
        errors = int(match.group(3))
        failed_test_num = failures + errors
        failed_tests = []
        if failures or errors:
            failed_tests = [test_name.split(".")[-1].replace("::", "#")]
        # reported failing test number and actual number of collected failing tests should match
        assert len(failed_tests) == failed_test_num
        return 0, failed_tests


def individual_run(repo_path, proj, bug_id, example_test):
    # test class generation & addition

    needed_elements = needed_imports_by_bug_id(repo_path, proj, bug_id, example_test)

    test_add_func = inject_test_by_bug_id

    test_name = test_add_func(repo_path, proj, bug_id, example_test, needed_elements)

    # actual running experiment
    fib_error_msg = None
    compile_status, compile_msg = compile_repo(repo_path)
    if compile_status != 0:
        status = -2
        failed_tests = []
    else:
        status, failed_tests = run_test(repo_path, test_name)

    return {
        'compile_error': status == -2,
        'runtime_error': status == -1,
        'failed_tests': failed_tests,
        'autogen_failed': len(failed_tests) > 0,
        'fib_error_msg': fib_error_msg,
        'compile_msg': compile_msg if status == -2 else None
    }


def twover_run_experiment(proj, bug_id, example_tests):
    """
    returns results in order of example_tests.
    """
    print(f'Running experiment for {proj}-{bug_id})')

    # init
    repo_path = inject_prefix_rootdir(proj, bug_id)
    test_dir = rwb_util.test_path_prefix(proj, bug_id)
    
    buggy_results = []
    fixed_results = []
    for example_test in tqdm(example_tests):
        print("Running experiment for buggy version")
        git_reset(repo_path+'b')
        git_clean(repo_path+'b')
        example_test = enforce_static_assertions(example_test)
        try:
            buggy_info = individual_run(repo_path+'b', proj, bug_id, example_test)

        except Exception as e:
            buggy_info = f'[error] {repr(e)}'
        
        print(buggy_info)
        buggy_results.append(buggy_info)
        
        print("Running experiment for fixed version")
        git_reset(repo_path+'f')
        git_clean(repo_path+'f')
        example_test = enforce_static_assertions(example_test)
        try:
            fixed_info = individual_run(repo_path+'f', proj, bug_id, example_test)

        except Exception as e:
            fixed_info = f'[error] {repr(e)}'

        print(fixed_info)
        fixed_results.append(fixed_info)        
    
    # Matching results together
    final_results = []
    for buggy_info, fixed_info in zip(buggy_results, fixed_results):
        if isinstance(buggy_info, str): # Test is syntactically incorrect (JavaSyntaxError)
            final_results.append(buggy_info)
            continue
        if isinstance(fixed_info, str): # Test is syntactically incorrect (JavaSyntaxError)
            final_results.append(fixed_info)
            continue

        fails_in_buggy_version = any(map(lambda x: 'AutoGen' in x, buggy_info['failed_tests']))
        fails_in_fixed_version = any(map(lambda x: 'AutoGen' in x, fixed_info['failed_tests']))

        success = (fails_in_buggy_version and not fails_in_fixed_version)

        final_results.append({
            'buggy': buggy_info,
            'fixed': fixed_info,
            'success': success,
        })
    return final_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_test_dir', default='../data/RWB/gen_tests/')
    parser.add_argument('--exp_name', default='RWB')
    args = parser.parse_args()

    GEN_TEST_DIR = args.gen_test_dir

    bug2tests = defaultdict(list)
    
    for gen_test_file in glob.glob(os.path.join(GEN_TEST_DIR, '*.txt')):
        bug_id = '_'.join(os.path.basename(gen_test_file).split('_')[:2])
        bug2tests[bug_id].append(gen_test_file)

    if os.path.exists(f'../results/{args.exp_name}.json'):
        exec_results = json.load(open(f'../results/{args.exp_name}.json'))
    else:  
        exec_results = {}
    task_args = []
    for bug_key, tests in tqdm(bug2tests.items()):
        if bug_key in exec_results.keys() and len(tests) == len(exec_results[bug_key]):
            print(bug_key)
            continue

        project, bug_id = bug_key.split('_')
        bug_id = int(bug_id)
        res_for_bug = {}

        example_tests = []
        for test_file in tests:
            with open(test_file, "r", encoding="utf-8") as f:
                test_content = f.read().strip()
            example_tests.append(test_content)

        task_args += [(project, bug_id, example_tests)]

        results = twover_run_experiment(project, bug_id, example_tests)

        if results is None:
            continue

        for test_path, res in zip(tests, results):
            res_for_bug[os.path.basename(test_path)] = res
        exec_results[bug_key] = res_for_bug

        break

        with open(f'../results/{args.exp_name}.json', 'w') as f:
            json.dump(exec_results, f, indent=4)