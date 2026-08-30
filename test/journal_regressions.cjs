const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '..', 'mainview.html'), 'utf8');

async function main() {
  const elements = new Map();
  const element = id => {
    if (!elements.has(id)) elements.set(id, {style: {}, hidden: true, replaceChildren() {}, appendChild() {}});
    return elements.get(id);
  };
  const context = vm.createContext({
    console, AbortController, setTimeout,
    document: {getElementById: element, querySelectorAll: () => [], createDocumentFragment: () => ({appendChild() {}})},
    createEventCard: image => image,
    currentEventDateFilter: '2026-08-30', currentEventCameraFilter: '', alertsOnlyFilter: false,
    currentImageSearchText: '', currentSimilarImage: null, uploadedImageData: null,
    is_face: false, eventStart: 0, hasMoreEvents: true, dontRefresh: false, EVENT_PAGE_SIZE: 10,
    fetch: async () => ({ok: true, json: async () => ({images: []})})
  });
  const journal = source.slice(source.indexOf('    let previousEventImages = null;'), source.indexOf('    async function loadMoreEventImages()'));
  vm.runInContext(journal, context);
  await vm.runInContext('loadEventImages()', context);
  assert.equal(context.dontRefresh, false, 'Normal journal refresh must resume');
  context.currentSimilarImage = 'sample.jpg';
  await vm.runInContext('loadEventImages()', context);
  assert.equal(context.currentSimilarImage, 'sample.jpg', 'Pagination must retain similar-image query');
  assert.equal(context.dontRefresh, true, 'Search results must not be replaced by polling');
  context.currentSimilarImage = null;
  context.fetch = async () => ({ok: false});
  await vm.runInContext('loadEventImages()', context);
  assert.equal(vm.runInContext('previousEventImages', context), null, 'Failed results must not poison recovery cache');
  assert.equal(element('journalRetry').hidden, false);
  context.fetch = async () => ({ok: true, json: async () => ({images: []})});
  await vm.runInContext('loadEventImages()', context);
  assert.equal(element('journalRetry').hidden, true);

  const video = {readyState: 0, currentTime: 999, play: () => Promise.resolve(), closest: () => null};
  const playback = vm.createContext({
    console, Map, Number, currentStreamDate: '2026-08-29', announce() {},
    document: {getElementById: id => id.startsWith('video-') ? video : {}, addEventListener() {}},
    restartAllStreams() {},
  });
  vm.runInContext(source.slice(source.indexOf('    const pendingReplay = new Map();'), source.indexOf('    async function playAllCamerasAtTime(')), playback);
  vm.runInContext("playVideoAtTime('test', 42, '2026-08-30')", playback);
  assert.equal(video.currentTime, 999, 'Do not seek the old recording before new metadata');
  video.readyState = 1;
  vm.runInContext("applyPendingReplay('test')", playback);
  assert.equal(video.currentTime, 42);
  vm.runInContext("playVideoAtTime('test', null, '2026-08-30')", playback);
  assert.equal(video.currentTime, 42, 'Missing footage must not seek to zero');
  console.log('Journal refresh, search pagination, error recovery, and delayed replay regressions passed.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
