---
layout: post
title:  "Audio Matching with Dejavu"
date:   2016-05-04 22:00:00 -0800
comments: true
tags: Dejavu, Audio Fingerprinting, Audio Matching, Song Matching, Shazam, SoundHound
---

Fingerprint Alignment
=====================
Songs can contain hundreds of fingerprints per second of audio. So how do we sort through all of the fingerprint matches to find the right song?

We could look at which song has the most matching fingerprints, but that would ignore the sequence of the fingerprints. Instead, we're actually going to look at the "relative offset" of the fingerprints, and we'll see that this approach captures all three possible aspects of alignment:

  1. How many fingerprints match
  2. The sequence of the fingerprints
  3. The spacing between fingerprints  

Here's how it works.

Let's look at four example fingerprints that show up in our sample clip; we'll call the fingerprints A, B, C and D. This illustration shows the locations of these four fingerprints both in our sample clip and in the original song.

![sample_clip_and_song]

All of the keypoints (A, B, C, D) are found in the original song, but they are at different offsets in the recorded clip versus the original song. 

<div class="message">
TODO - I'm just using small integers for the offsets here. In practice, the offset values are in terms of ?? (samples?)
</div>

So how do we align them? If you calculate the difference between the offset values, you’ll notice that the "diff" is the same for all of the matching keypoints!

![offset_diffs]

This "Offset Diff" value is actually the exact point in the original song at which we started recording our sample. So you could call it the "Sample Start Time" if you'd like.

In fact, each fingerprint match is actually a vote for a particular song and a particular sample start time within that song. To find the correct song, you tally up all of the votes and select the song (and sample start time) with the highest number of votes.

Let's extend our initial example to include a second song which also contains some of the fingerprints in our sample.

![sample_clip_and_two_songs]

Here are the list of fingerprint matches, and their calculated "offset diff"s.
![offset_diffs_two_songs]

To pick the correct song, we use a data structure to help us count up the votes. We have a table of "diff" values, each of which maps to a table of songs and their total number of votes.
![voting]

As we tally up the votes, we can keep track of which (song, offset diff) pair has the highest tally. Once all of the matches are tallied up, we have our winner.


[sample_clip_and_song]: {{ site.url }}/assets/Dejavu/Sample_Clip_And_Original_Song.png
[offset_diffs]: {{ site.url }}/assets/Dejavu/Offset_Diffs.png
[sample_clip_and_two_songs]: {{ site.url }}/assets/Dejavu/sample_clip_and_two_songs.png
[offset_diffs_two_songs]: {{ site.url }}/assets/Dejavu/offset_diffs_two_songs.png
[voting]: {{ site.url }}/assets/Dejavu/voting.png