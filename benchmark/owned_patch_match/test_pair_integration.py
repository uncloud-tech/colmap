# SPDX-License-Identifier: BSD-3-Clause
"""CPU/source tests only, never CUDA ordering or numerical qualification."""
import pathlib
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
MVS = ROOT / 'src/colmap/mvs'


def function(text, signature):
    start = text.index(signature)
    begin = text.index('{', start)
    depth = 1
    end = begin + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


class PairIntegration(unittest.TestCase):
    def test_owned_stop_is_error_but_stock_stop_returns(self):
        text = (MVS / 'patch_match.cc').read_text()
        body = function(text, 'void PatchMatchController::ProcessProblem(')
        prefix = body[body.index('{')+1:body.index('  const auto& model')]
        cpp = r'''
#include <cassert>
#include <stdexcept>
struct Check { bool ok;explicit Check(bool b):ok(b){}~Check() noexcept(false){if(!ok)throw std::runtime_error("stopped");}template<class T>Check& operator<<(const T&){return *this;} };
#define THROW_CHECK(x) Check(bool(x))
struct Controller {bool owned_pair_=false,reached=false;bool CheckIfStopped(){return true;}
void Run(){
''' + prefix + r'''
reached=true;}};
int main(){Controller stock;stock.Run();assert(!stock.reached);Controller pair;pair.owned_pair_=true;
bool rejected=false;try{pair.Run();}catch(const std::exception&){rejected=true;}assert(rejected&&!pair.reached);}
'''
        with tempfile.TemporaryDirectory(prefix='owned-stop-') as tmp:
            source=pathlib.Path(tmp)/'test.cc';exe=pathlib.Path(tmp)/'test';source.write_text(cpp)
            p=subprocess.run(['g++','-std=c++17',str(source),'-o',str(exe)],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            p=subprocess.run([str(exe)],capture_output=True,text=True,timeout=10)
            self.assertEqual(p.returncode,0,p.stderr)

    def test_pair_owner_scope_and_sweep_tag_seams(self):
        text = (MVS / 'patch_match.cc').read_text()
        self.assertIn('void PatchMatchController::RunOwnedPairJob(', text)
        self.assertIn('owned_pair_->ReadyAndWait(owned_pair_job_)', text)
        self.assertIn('owned_pair_->AwaitOutputs(owned_pair_job_)', text)
        self.assertIn('owned_pair_->Cancel(owned_pair_job_, std::current_exception())', text)
        cuda = (MVS / 'patch_match_cuda_owned.cu').read_text()
        self.assertIn('sweep_observer_(iter + 1, sweep + 1)', cuda)
        self.assertLess(cuda.index('sweep_timer.Print('), cuda.index('sweep_observer_(iter + 1, sweep + 1)'))

    def test_driver_compiles_against_fake_host_interfaces(self):
        with tempfile.TemporaryDirectory(prefix='owned-pair-driver-syntax-') as tmp:
            fake = pathlib.Path(tmp)
            files = {
                'colmap/controllers/option_manager.h': r'''
#pragma once
#include "colmap/mvs/patch_match_options.h"
#include <filesystem>
#include <memory>
#include <string>
namespace colmap {
struct BaseOptionManager { explicit BaseOptionManager(bool=true){} void Reset(bool){}
 template<class T> void AddRequiredOption(const std::string&,T*){}
 template<class T> void AddDefaultOption(const std::string&,T*){}
 bool Parse(int,char**){return true;} };
struct OptionManager:BaseOptionManager { explicit OptionManager(bool b):BaseOptionManager(b){}
 std::shared_ptr<mvs::PatchMatchOptions> patch_match_stereo=std::make_shared<mvs::PatchMatchOptions>();
 void AddPatchMatchStereoOptions(){} bool Read(const std::filesystem::path&,bool){return true;} };
}
''',
                # The production driver includes this to make Workspace
                # complete; keep this unrelated host-interface test isolated.
                'colmap/mvs/workspace.h': '#pragma once\n',
                'colmap/mvs/patch_match.h': r'''
#pragma once
#include "colmap/mvs/patch_match_options.h"
#include "colmap/mvs/owned_pair_coordinator.h"
#include <filesystem>
#include <functional>
#include <vector>
namespace colmap::mvs {
struct Image {size_t GetWidth()const{return 1;}size_t GetHeight()const{return 1;}
 const std::filesystem::path& GetPath()const{static std::filesystem::path p;return p;}};
struct PatchMatch {struct Problem{int ref_image_idx;std::vector<int> src_image_idxs;std::vector<Image>* images;};};
enum class PatchMatchExecution{kOwnedCuda};
struct PatchMatchController {
 struct OwnedPairHooks {
 std::function<void(size_t,const PatchMatchOptions&,const PatchMatch::Problem&)> check_inputs,metadata;
 std::function<void(size_t,int,int)> sweep;std::function<void(size_t,const char*)> output_phase;
 };
 PatchMatchController(const PatchMatchOptions&,const std::filesystem::path&,const std::string&,const std::string&,const std::filesystem::path&,PatchMatchExecution){}
 void RunOwnedPairJob(owned::PairCoordinator&,size_t,const OwnedPairHooks&){}
};
}
''',
                'colmap/util/logging.h': '#pragma once\n#include <sstream>\n#define LOG(x) std::ostringstream()\nnamespace colmap {inline void InitializeGlog(char**) {}}\n',
                'colmap/util/file.h': '#pragma once\n#include <filesystem>\n#include <string>\n#include <vector>\nnamespace colmap {inline std::vector<std::string> ReadTextFileLines(const std::filesystem::path&){return {};}}\n',
                'colmap/util/string.h': '#pragma once\n#include <string>\nnamespace colmap {inline void StringTrim(std::string*){} inline void StringToLower(std::string*){}}\n',
            }
            for name, data in files.items():
                path = fake / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data)
            result = subprocess.run(['g++', '-std=c++17', '-pthread', '-Wall', '-Wextra', '-Werror',
                '-DCOLMAP_GLOG_HAS_COLOR_SUPPORT=1', '-fsyntax-only', '-I', tmp, '-I', str(ROOT/'src'),
                str(ROOT/'src/colmap/exe/owned_pair.cc')], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            # Execute the ACTUAL filesystem validation function. Only option/
            # controller dependencies are fake; filesystem operations are real.
            (fake/'colmap/util/file.h').write_text(r'''
#pragma once
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace colmap {inline std::vector<std::string> ReadTextFileLines(const std::filesystem::path& p){std::ifstream f(p);std::vector<std::string> out;std::string s;while(std::getline(f,s))out.push_back(s);return out;}}
''')
            source=fake/'filesystem.cc'
            source.write_text('#define main original_driver_main\n#include "'+str(ROOT/'src/colmap/exe/owned_pair.cc')+'"\n#undef main\n'+r'''
#include <cassert>
#include <fstream>
namespace fs=std::filesystem;
void Make(const fs::path& root,bool alias){
 for(const char* part:{"images","sparse","stereo/depth_maps"})fs::create_directories(root/part);
 if(alias)fs::create_directory_symlink(root/"stereo/depth_maps",root/"stereo/normal_maps");
 else fs::create_directory(root/"stereo/normal_maps");
 std::ofstream(root/"images/ref.jpg")<<"fixture";
 for(const char* file:{"cameras.txt","images.txt","points3D.txt"})std::ofstream(root/"sparse"/file)<<"fixture";
 std::ofstream(root/"stereo/patch-match.cfg")<<"ref.jpg\nsource.jpg\n";
}
bool Valid(const fs::path& root){
 colmap::JobOptions job;job.workspace=root;job.format="colmap";
 job.patch_match_stereo->geom_consistency=false;job.patch_match_stereo->gpu_index="0";
 job.patch_match_stereo->cache_size=.5;
 try{colmap::ValidateJob(job,{3ULL<<30,8ULL<<30});return true;}catch(const std::exception&){return false;}
}
int main(int argc,char** argv){assert(argc==2);const fs::path root(argv[1]);
 Make(root/"valid",false);assert(Valid(root/"valid"));
 Make(root/"alias",true);assert(!Valid(root/"alias"));
 Make(root/"partial",false);std::ofstream(root/"partial/stereo/depth_maps/ref.jpg.photometric.bin")<<"partial";assert(!Valid(root/"partial"));
 Make(root/"dangling",false);fs::create_symlink(root/"absent",root/"dangling/stereo/depth_maps/ref.jpg.photometric.bin");assert(!Valid(root/"dangling"));
}
''')
            exe=fake/'filesystem-test'
            result=subprocess.run(['g++','-std=c++17','-pthread','-I',tmp,'-I',str(ROOT/'src'),str(source),'-o',str(exe),'-lOpenImageIO','-lOpenImageIO_Util'],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            result=subprocess.run([str(exe),str(fake/'fixtures')],capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_real_owned_prepare_run_host_methods(self):
        text = (MVS / 'patch_match.cc').read_text()
        bodies = '\n'.join(function(text, name) for name in (
            'void PatchMatch::PrepareOwned()',
            'void PatchMatch::RunPreparedOwned()',
            'void PatchMatch::Run()'))
        source = r'''
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cassert>
struct Check { bool ok; explicit Check(bool b):ok(b){} ~Check() noexcept(false){if(!ok)throw std::runtime_error("check");} template<class T> Check& operator<<(const T&){return *this;} };
#define THROW_CHECK(x) ::Check(bool(x))
#define THROW_CHECK_NOTNULL(x) ::Check((x)!=nullptr)
#define LOG(x) std::ostringstream()
#define LOG_HEADING2(x) ((void)0)
enum class PatchMatchExecution { kStock, kOwnedCuda };
struct Problem { int ref_image_idx=3; };
int constructed=0,ran=0;bool fail=false;
struct PatchMatchCuda { PatchMatchCuda(int,const Problem&){++constructed;} void Run(){++ran;if(fail)throw std::runtime_error("backend");} };
namespace owned {using PatchMatchCuda=::PatchMatchCuda;}
struct PatchMatch {
  int options_=0; Problem problem_;
  PatchMatchExecution execution_=PatchMatchExecution::kOwnedCuda;
  bool owned_run_started_=false,owned_run_completed_=false;
  std::unique_ptr<PatchMatchCuda> patch_match_cuda_;
  std::unique_ptr<owned::PatchMatchCuda> owned_patch_match_cuda_;
  void Check()const{}
  void PrepareOwned();void RunPreparedOwned();void Run();
};
''' + bodies + r'''
template<class F> bool rejects(F f){try{f();return false;}catch(const std::exception&){return true;}}
int main(){
  PatchMatch a;assert(rejects([&]{a.RunPreparedOwned();}));
  a.PrepareOwned();assert(constructed==1&&ran==0);
  assert(rejects([&]{a.PrepareOwned();}));
  a.RunPreparedOwned();assert(ran==1&&a.owned_run_completed_);
  assert(rejects([&]{a.RunPreparedOwned();}));
  PatchMatch b;b.Run();assert(constructed==2&&ran==2&&b.owned_run_completed_);
  PatchMatch c;c.PrepareOwned();fail=true;
  assert(rejects([&]{c.RunPreparedOwned();}));assert(!c.owned_run_completed_);
  assert(rejects([&]{c.RunPreparedOwned();}));fail=false;
  PatchMatch stock;stock.execution_=PatchMatchExecution::kStock;
  assert(rejects([&]{stock.PrepareOwned();}));stock.Run();
  assert(stock.patch_match_cuda_&&!stock.owned_patch_match_cuda_);
}
'''
        with tempfile.TemporaryDirectory(prefix='owned-pair-methods-') as tmp:
            cpp = pathlib.Path(tmp) / 'test.cc'
            exe = pathlib.Path(tmp) / 'test'
            cpp.write_text(source)
            result = subprocess.run(['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
                                     str(cpp), '-o', str(exe)], capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([str(exe)], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
