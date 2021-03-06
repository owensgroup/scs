function z = proj_dual_cone(z,K) %% DUAL CONE
% lp cone
z(K.f+1:K.l+K.f) = max(z(K.f+1:K.l+K.f),0);
idx=K.l+K.f;
% SOCs
for i=1:length(K.q)
    z(idx+1:idx+K.q(i)) = proj_soc(z(idx+1:idx+K.q(i)));
    idx=idx+K.q(i);
end
% SDCs
for i=1:length(K.s)
    z(idx+1:idx+K.s(i)^2) = proj_sdp(z(idx+1:idx+K.s(i)^2),K.s(i));
    idx=idx+K.s(i)^2;
end
% EXP cones
for i=1:K.ep
    z(idx+1:idx+3) = z(idx+1:idx+3) + proj_exp(-z(idx+1:idx+3));
    idx=idx+3;
end
% dual EXP cones
for i=1:K.ed
    z(idx+1:idx+3) = proj_exp(z(idx+1:idx+3));
    idx=idx+3;
end
end