# SPDX-License-Identifier: BSD-3-Clause
"""Real C++ coordinator/owner-runner/event sink, fake resources; CPU only."""
import pathlib
import subprocess
import tempfile
import unittest
ROOT=pathlib.Path(__file__).resolve().parents[2]

CPP=r'''
#include "colmap/mvs/owned_pair_events.h"
#include "colmap/mvs/owned_pair_run.h"
#include <atomic>
#include <cassert>
#include <iostream>
using namespace colmap::mvs::owned;
struct Resource { std::atomic<int>& alive;explicit Resource(std::atomic<int>& a):alive(a){++alive;}~Resource(){--alive;} };
int main(int argc,char** argv){
 assert(argc==2);std::string mode=argv[1];bool spawn_failure=mode=="spawn_failure";
 if(mode=="pair")assert(write(STDOUT_FILENO,"unrelated partial",17)==17);
 PairEvents e(mode=="pair"?"pair":"serial");
 PairCoordinator c(mode=="pair"?PairDispatch::kPair:PairDispatch::kSerial,
 {9,9},{1,1},{{{1,1},{1,1}}},std::chrono::seconds(3),[&](const PairEvent& p){e.Coordinator(p);});
 std::atomic<int> alive{0};int starts=0;
 auto factory=[&](std::function<void()> f){if(spawn_failure&&starts++==1)throw std::runtime_error("spawn failure");return std::thread(std::move(f));};
 try {
 RunPairOwners(c,[&](size_t job){
  e.Owner(job);c.BeginPreparation(job);Resource r(alive);e.Identity(job,7+job,0);c.ReadyAndWait(job);assert(alive==2);
  for(int i=1;i<=5;++i)for(int s=1;s<=4;++s)e.Emit(job,"sweep_complete","iteration="+std::to_string(i)+" sweep="+std::to_string(s));
  c.SolverDone(job);c.BeginOutput(job);
  for(auto phase:{"depth_begin","depth_end","normal_begin","normal_end"})e.Emit(job,phase);
  c.EndOutput(job);c.AwaitOutputs(job);
 },[&](size_t job,const char* phase){e.Emit(job,phase);},factory);
 assert(!spawn_failure);assert(alive==0);assert(c.ReservedBytes().host==0);e.Finished();
 }catch(const std::runtime_error& ex){
  assert(spawn_failure);assert(std::string(ex.what())=="spawn failure");assert(alive==0);
  assert(c.ReservedBytes().host==0);e.Emit(PairEvents::kBatch,"batch_failed","status=failure");
 }
}
'''

class PairEvents(unittest.TestCase):
    def test_two_modes_and_thread_creation_failure(self):
        with tempfile.TemporaryDirectory(prefix='owned-pair-events-') as tmp:
            source=pathlib.Path(tmp)/'test.cc';binary=pathlib.Path(tmp)/'test';source.write_text(CPP)
            p=subprocess.run(['g++','-std=c++17','-pthread','-Wall','-Wextra','-Werror','-I',str(ROOT/'src'),str(source),'-o',str(binary)],capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stderr)
            for mode in ('serial','pair','spawn_failure'):
                with self.subTest(mode=mode):
                    p=subprocess.run([str(binary),mode],capture_output=True,text=True,timeout=10)
                    self.assertEqual(p.returncode,0,p.stderr)
                    lines=[line for line in p.stdout.splitlines() if 'OWNED_PAIR_EVENT' in line]
                    self.assertTrue(lines)
                    records=[]
                    for line in lines:
                        self.assertTrue(line.startswith('OWNED_PAIR_EVENT '),line)
                        fields=dict(item.split('=',1) for item in line.split()[1:])
                        self.assertEqual(fields['schema'],'1')
                        records.append(fields)
                    self.assertEqual([int(x['seq']) for x in records],list(range(1,len(records)+1)))
                    self.assertEqual(len({x['batch'] for x in records}),1)
                    times=[int(x['monotonic_ns']) for x in records]
                    self.assertEqual(times,sorted(times))
                    events=[x['event'] for x in records]
                    self.assertLess(events.index('owners_joined'),events.index('ledger_release'))
                    self.assertEqual(events.count('ledger_release'),1)
                    if mode=='spawn_failure':
                        self.assertNotIn('batch_finished',events)
                        self.assertNotIn('solver_start',events)
                        self.assertEqual(events[-1],'batch_failed')
                        continue
                    self.assertEqual(events[-1],'batch_finished')
                    for job in ('A','B'):
                        selected=[x for x in records if x['job']==job]
                        self.assertEqual(sum(x['event']=='sweep_complete' for x in selected),20)
                        ready=next(x for x in selected if x['event']=='ready')
                        self.assertGreater(int(ready['owner_tid']),0)
                        self.assertEqual(ready['owner_tid'],ready['emitter_tid'])
                        names=[x['event'] for x in selected]
                        self.assertLess(names.index('depth_end'),names.index('normal_begin'))
                        self.assertLess(names.index('cleanup_complete'),names.index('retired'))
                    logical=sorted((x for x in records if 'coord_seq' in x),key=lambda x:int(x['coord_seq']))
                    both=next(int(x['coord_seq']) for x in logical if x['event']=='both_ready')
                    for x in logical:
                        if x['event']=='solver_start':self.assertGreater(int(x['coord_seq']),both)
                    if mode=='serial':
                        end=next(int(x['coord_seq']) for x in logical if x['job']=='A' and x['event']=='solver_end')
                        start=next(int(x['coord_seq']) for x in logical if x['job']=='B' and x['event']=='solver_start')
                        self.assertGreater(start,end)

if __name__=='__main__':unittest.main()
