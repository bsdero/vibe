10 REM Quick Sort in GW-BASIC
20 DIM A%(100) ' Adjust array size as needed
30 INPUT "Enter number of elements: ", N
40 IF N > 100 THEN PRINT "Array too large!"; GOTO 30
50 PRINT "Enter the elements:"
60 FOR I = 1 TO N
70   INPUT A%(I)
80 NEXT I
90 GOSUB 1000 ' Call quicksort subroutine
100 PRINT "Sorted array:"
110 FOR I = 1 TO N
120   PRINT A%(I);
130 NEXT I
140 END

1000 REM Quicksort subroutine
1010 ' Parameters: A%() - array, L - low index, H - high index
1020 IF L >= H THEN RETURN ' Base case: subarray with 0 or 1 element
1030 I = L
1040 J = H
1050 PIVOT = A%((L + H) \ 2) ' Choose middle element as pivot
1060 WHILE I <= J
1070   WHILE A%(I) < PIVOT AND I <= J
1080     I = I + 1
1090   WEND
1100   WHILE A%(J) > PIVOT AND I <= J
1110     J = J - 1
1120   WEND
1130   IF I <= J THEN
1140     SWAP A%(I), A%(J)
1150     I = I + 1
1160     J = J - 1
1170   ENDIF
1180 WEND
1190 IF L < J THEN GOSUB 1000 ' Recursive call on left partition
1200 IF I < H THEN GOSUB 1000 ' Recursive call on right partition
1210 RETURN
